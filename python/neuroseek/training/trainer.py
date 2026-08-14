"""Detached training entrypoint. It records only observed process/model values."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import signal
import sys
import time
import traceback
import uuid
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from neuroseek.models.policy import LocalCandidateSubgraph, NavigatorPolicy, OP_NAMES
from neuroseek.rl.ppo import ppo_update
from neuroseek.telemetry.events import EventWriter
from neuroseek.training.checkpoint import has_checkpoint_candidates, load_checkpoint, load_latest, publish_latest, prune_periodic, save_atomic
from neuroseek.telemetry.jetson import snapshot as telemetry_snapshot
from neuroseek.telemetry.jetson import JetsonTelemetryCollector, ThermalSafetyPolicy

STOP = False


def _stop(_sig: int, _frame: object) -> None:
    global STOP
    STOP = True


def _read_toml(path: Path) -> dict:
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib
    with path.open("rb") as source:
        return tomllib.load(source)


def _sha256_file(path: Path) -> str:
    """Return a byte-level provenance binding for immutable run inputs."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seed(value: int) -> None:
    random.seed(value); np.random.seed(value); torch.manual_seed(value)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(value)


def _state(device: torch.device, batch: int, dim: int) -> torch.Tensor:
    # Synthetic state is permitted only for smoke; production task adapter replaces it.
    return torch.randn(batch, dim, device=device)


def _phase_for_elapsed(config: dict, elapsed_seconds: float, budget_seconds: float) -> str:
    """Resolve a wall-clock curriculum phase without episode-count drift.

    Full mode supplies an explicit ordered TOML ``[[phase]]`` table.  The
    compact smoke/trial configurations retain the conservative BC-then-PPO
    fallback so their bounded acceptance tests do not need production timing.
    """
    phases = config.get("phase", [])
    if not phases:
        return "behavior_cloning" if elapsed_seconds < budget_seconds * 0.10 else "graph_ppo"
    elapsed = max(0.0, elapsed_seconds)
    cursor = 0.0
    for item in phases:
        name = str(item.get("name", ""))
        duration = float(item.get("seconds", 0.0))
        if not name or duration <= 0.0:
            raise ValueError("each [[phase]] requires a nonempty name and positive seconds")
        cursor += duration
        if elapsed < cursor:
            return name
    return str(phases[-1]["name"])


def _validate_phase_schedule(config: dict, mode: str, budget_seconds: float) -> None:
    phases = config.get("phase", [])
    if mode == "full" and not phases:
        raise ValueError("full mode requires an explicit wall-clock [[phase]] schedule")
    if phases:
        total = sum(float(item.get("seconds", 0.0)) for item in phases)
        if abs(total - budget_seconds) > 1e-6:
            raise ValueError(f"phase schedule {total:g}s does not equal run budget {budget_seconds:g}s")


def _minimum_free_disk_bytes(config: dict, mode: str) -> int:
    """Return the configured runtime disk reserve without imposing it on smoke."""
    raw = config.get("storage", {}).get("minimum_free_gib", 24.0 if mode == "full" else 0.0)
    value = float(raw)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("storage.minimum_free_gib must be finite and non-negative")
    return int(value * 1024**3)


def _navigator_local_scores(model: NavigatorPolicy, graph: object, env: object, candidates: list[int],
                            device: torch.device, candidate_cap: int = 256) -> tuple[list[int], torch.Tensor]:
    """Run the relational navigator only on a bounded real local subgraph.

    Candidate IDs and parent links were produced by the graph runtime.  This
    function intentionally never creates an all-entity tensor: it converts at
    most ``candidate_cap`` current frontier nodes plus their immediate parents
    into compact PyTorch tensors.  The returned score order is exactly the
    supplied bounded candidate prefix, which lets the strategist use the GNN
    for `TOPK` while retaining an auditable host cap on an 8 GB device.
    """
    selected = candidates[:candidate_cap]
    if not selected:
        raise ValueError("navigator requires a non-empty frontier")
    parents = env.parents[-1] if env.parents else {}
    parent_relations = env.parent_relations[-1] if env.parent_relations else {}
    node_ids = sorted(set(selected).union(parents.get(node, env.query.source) for node in selected))
    positions = {node: index for index, node in enumerate(node_ids)}
    observation = np.asarray(env.observation(), dtype=np.float32)
    features = np.repeat(observation[None, :], len(node_ids), axis=0)
    # These measured mmap features are cheap and do not infer semantics from
    # strings: compact ID, outgoing degree, and current-candidate bit.
    entity_count = max(1, int(graph.manifest.entity_count) - 1)
    for index, node in enumerate(node_ids):
        degree = int(graph.forward_offsets[node + 1] - graph.forward_offsets[node])
        features[index, 20] = node / entity_count
        features[index, 21] = min(np.log1p(degree) / 16.0, 1.0)
        features[index, 22] = float(node in selected)
    edge_source: list[int] = []
    edge_target: list[int] = []
    relations: list[int] = []
    for child in selected:
        parent = parents.get(child)
        if parent is not None and parent in positions:
            edge_source.append(positions[parent]); edge_target.append(positions[child])
            relations.append(parent_relations.get(child, 0))
    edges = torch.as_tensor([edge_source, edge_target], dtype=torch.long, device=device)
    rel_tensor = torch.as_tensor(relations, dtype=torch.long, device=device)
    candidate_indices = torch.as_tensor([positions[node] for node in selected], dtype=torch.long, device=device)
    subgraph = LocalCandidateSubgraph(
        node_features=torch.as_tensor(features, dtype=torch.float32, device=device),
        edge_index=edges, edge_relations=rel_tensor, candidate_indices=candidate_indices,
    )
    query = torch.as_tensor(observation, dtype=torch.float32, device=device)
    return selected, model.score_local_candidates(query, subgraph)


def _navigator_ranker(model: NavigatorPolicy, graph: object, device: torch.device, candidate_cap: int):
    """Create an inference-only ranker for the environment's real TOPK op."""
    def rank(env: object, candidates: list[int]) -> list[int]:
        selected, scores = _navigator_local_scores(model, graph, env, candidates, device, candidate_cap)
        values = scores.detach().float().cpu().numpy()
        ranked = [node for _score, node in sorted(zip(values.tolist(), selected), key=lambda item: (-item[0], item[1]))]
        # Nodes above the explicit local GNN cap remain visible to the search
        # VM in deterministic ID order; no candidate disappears silently.
        return ranked + candidates[len(selected):]
    return rank


def _collect_real_rollouts(model: NavigatorPolicy, graph: object, generator: object, device: torch.device, episodes: int, cuda_session: object, max_cuda_expand_edges: int, navigator_candidate_cap: int, *, episode_factory: object | None = None, semantic_search: object | None = None, hardware_cost_predictor: object | None = None, action_temperature: float = 1.0, latency_penalty_coefficient: float = 0.01, instruction_penalty: float = 0.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, object]]:
    """Execute sampled NEURO-ISA programs against real mmap graph episodes."""
    from neuroseek.search.environment import GraphSearchEnv
    states: list[torch.Tensor] = []; actions: list[torch.Tensor] = []; old_logprob: list[torch.Tensor] = []; returns: list[torch.Tensor] = []
    results = []
    trace_rows: list[tuple[object, object]] = []
    predicted_latencies_ms: list[float] = []
    model_latency_penalties: list[float] = []
    instruction_penalties: list[float] = []
    latency_penalties: list[float] = []
    for _ in range(episodes):
        query, proof = episode_factory() if episode_factory is not None else generator.next()
        env = GraphSearchEnv(graph, query, proof, cuda_session=cuda_session, max_cuda_expand_edges=max_cuda_expand_edges,
                             candidate_ranker=_navigator_ranker(model, graph, device, navigator_candidate_cap), semantic_search=semantic_search)
        episode: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]] = []
        for _instruction in range(len(query.relations) + 6):
            state = torch.as_tensor(env.observation(), device=device).unsqueeze(0)
            with torch.no_grad(): logits, _ = model(state); distribution = torch.distributions.Categorical(logits=logits / action_temperature); action = distribution.sample(); logprob = distribution.log_prob(action)
            result = env.step(int(action.item()))
            episode.append((state.squeeze(0), action.squeeze(0), logprob.squeeze(0), result.reward))
            if result.done: break
        if not result.done:
            result = env.step(11)  # bounded environment always closes the trace
            prior = episode[-1]; episode[-1] = (prior[0], prior[1], prior[2], prior[3] + result.reward)
        predicted_ms = None
        if hardware_cost_predictor is not None:
            predicted_ms = float(hardware_cost_predictor(result))
            if not np.isfinite(predicted_ms) or predicted_ms < 0:
                raise FloatingPointError("hardware cost predictor emitted an invalid latency")
        # The cost model is trained from observed CUDA probes.  Pair it with a
        # small explicit instruction cost so the policy optimizes the actual
        # dominant end-to-end cost: repeated host-side policy/VM round trips.
        model_latency_penalty = latency_penalty_coefficient * predicted_ms if predicted_ms is not None else 0.0
        per_instruction_penalty = instruction_penalty * len(result.trace)
        penalty = model_latency_penalty + per_instruction_penalty
        if penalty:
            prior = episode[-1]
            episode[-1] = (prior[0], prior[1], prior[2], prior[3] - penalty)
            result = type(result)(**{**result.__dict__, "reward": result.reward - penalty})
        if predicted_ms is not None:
            predicted_latencies_ms.append(predicted_ms)
        model_latency_penalties.append(model_latency_penalty)
        instruction_penalties.append(per_instruction_penalty)
        latency_penalties.append(penalty)
        running = 0.0
        for state, action, logprob, reward in reversed(episode):
            running = reward + 0.99 * running
            states.append(state); actions.append(action); old_logprob.append(logprob); returns.append(torch.tensor(running, device=device))
        results.append(result)
        trace_rows.append((query, result))
    operator_counts = {name: 0 for name in OP_NAMES}
    for _query, result in trace_rows:
        for token in result.trace:
            operation = str(token).split("(", 1)[0]
            if operation in operator_counts:
                operator_counts[operation] += 1
    representative_query, representative = trace_rows[-1]
    metrics = {"success_rate": float(np.mean([r.answer_correct for r in results])), "proof_validity": float(np.mean([r.valid_proof for r in results])), "nodes_per_query": float(np.mean([r.nodes_visited for r in results])), "edges_per_query": float(np.mean([r.edges_examined for r in results])), "credits_per_query": float(np.mean([r.credits for r in results])), "ann_calls_per_query": float(np.mean([r.ann_calls for r in results])), "ann_vectors_examined_per_query": float(np.mean([r.ann_vectors_examined for r in results])), "cuda_expansions_per_query": float(np.mean([r.cuda_expansions for r in results])), "navigator_ranked_candidates_per_query": float(np.mean([r.navigator_ranked_candidates for r in results])), "reward": float(np.mean([r.reward for r in results])),
               "live_search_task": str(representative_query.task_id), "live_search_family": str(representative_query.family),
               "live_search_trace": " -> ".join(str(token) for token in representative.trace),
               "live_search_result": "VALID" if representative.valid_proof else ("ANSWER_UNVERIFIED" if representative.answer_correct else "NO_ANSWER"),
               "operator_distribution": " ".join(f"{name}:{count}" for name, count in operator_counts.items() if count),
               "instructions_per_query": float(np.mean([len(result.trace) for result in results])),
               # A rejected model is reported as unavailable, never as a
               # fabricated 0 ms prediction.  The explicit instruction cost
               # remains a real, transparent low-latency objective.
               "predicted_latency_ms_per_query": float(np.mean(predicted_latencies_ms)) if predicted_latencies_ms else None,
               "model_latency_penalty_per_query": float(np.mean(model_latency_penalties)) if predicted_latencies_ms else None,
               "instruction_penalty_per_query": float(np.mean(instruction_penalties)),
               "latency_penalty_per_query": float(np.mean(latency_penalties))}
    return torch.stack(states), torch.stack(actions), torch.stack(old_logprob), torch.stack(returns), metrics


def _behavior_clone(model: NavigatorPolicy, optimizer: torch.optim.Optimizer, graph: object, generator: object, device: torch.device, episodes: int, cuda_session: object, max_cuda_expand_edges: int, navigator_candidate_cap: int, *, episode_factory: object | None = None, semantic_search: object | None = None) -> dict[str, object]:
    from neuroseek.search.environment import GraphSearchEnv
    states: list[np.ndarray] = []; targets: list[int] = []; cuda_counts: list[int] = []
    navigator_losses: list[torch.Tensor] = []
    results: list[object] = []
    trace_rows: list[tuple[object, object]] = []
    for _ in range(episodes):
        query, proof = episode_factory() if episode_factory is not None else generator.next(); env = GraphSearchEnv(graph, query, proof, cuda_session=cuda_session, max_cuda_expand_edges=max_cuda_expand_edges, semantic_search=semantic_search)
        for action in env.demonstration():
            states.append(env.observation()); targets.append(action); result = env.step(action)
            # Supervise the local GNN against the known next proof node after
            # its first real relation expansion.  The policy's graph runtime
            # still supplies all candidates; this is not a global-label head.
            if action == 2 and len(env.parents) == 1 and len(proof) > 1 and proof[1] in env.frontier:
                candidates, scores = _navigator_local_scores(model, graph, env, sorted(env.frontier), device, navigator_candidate_cap)
                # The local scorer has a strict candidate cap.  A valid proof
                # child can therefore be in the full runtime frontier while
                # falling outside this bounded training view.  It has no
                # representable local class label in that case: retain the
                # strategist demonstration and skip only this auxiliary GNN
                # supervision instead of terminating the detached run.
                if proof[1] in candidates:
                    target_index = candidates.index(proof[1])
                    navigator_losses.append(torch.nn.functional.cross_entropy(scores.unsqueeze(0), torch.tensor([target_index], device=device)))
            if result.done: break
        cuda_counts.append(result.cuda_expansions)
        results.append(result)
        trace_rows.append((query, result))
    x = torch.as_tensor(np.stack(states), device=device); y = torch.as_tensor(targets, device=device)
    logits, _ = model(x); strategist_loss = torch.nn.functional.cross_entropy(logits, y)
    navigator_loss = torch.stack(navigator_losses).mean() if navigator_losses else torch.zeros((), device=device)
    loss = strategist_loss + 0.25 * navigator_loss
    if not torch.isfinite(loss): raise FloatingPointError("non-finite behavior cloning loss")
    optimizer.zero_grad(set_to_none=True); loss.backward(); grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)); optimizer.step()
    representative_query, representative = trace_rows[-1]
    operator_counts = {name: 0 for name in OP_NAMES}
    for _query, episode_result in trace_rows:
        for token in episode_result.trace:
            operation = str(token).split("(", 1)[0]
            if operation in operator_counts:
                operator_counts[operation] += 1
    return {"policy_loss": float(loss.detach()), "navigator_loss": float(navigator_loss.detach()), "value_loss": 0.0, "entropy": float(torch.distributions.Categorical(logits=logits).entropy().mean().detach()), "kl": 0.0, "gradient_norm": grad,
            "reward": float(np.mean([item.reward for item in results])), "success_rate": float(np.mean([item.answer_correct for item in results])),
            "proof_validity": float(np.mean([item.valid_proof for item in results])), "nodes_per_query": float(np.mean([item.nodes_visited for item in results])),
            "edges_per_query": float(np.mean([item.edges_examined for item in results])), "credits_per_query": float(np.mean([item.credits for item in results])),
            "cuda_expansions_per_query": float(np.mean(cuda_counts)), "navigator_ranked_candidates_per_query": float(np.mean([item.navigator_ranked_candidates for item in results])),
            "live_search_task": str(representative_query.task_id), "live_search_family": str(representative_query.family),
            "live_search_trace": " -> ".join(str(token) for token in representative.trace),
            "live_search_result": "VALID" if representative.valid_proof else "DEMONSTRATION_INVALID",
            "operator_distribution": " ".join(f"{name}:{count}" for name, count in operator_counts.items() if count)}


def _checkpoint(model: NavigatorPolicy, optimizer: torch.optim.Optimizer, run_dir: Path, phase: str, step: int, best: float, config: dict, elapsed_seconds: float, *, generator: object | None = None, phase_state: dict | None = None, provenance: dict[str, str] | None = None) -> Path:
    payload = {"format": 1, "global_step": step, "phase": phase, "best_metric": best, "elapsed_seconds": elapsed_seconds, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "rng_python": random.getstate(), "rng_numpy": np.random.get_state(), "rng_torch": torch.get_rng_state(), "config": config, "saved_at": time.time()}
    if generator is not None:
        state_dict = getattr(generator, "state_dict", None)
        if not callable(state_dict):
            raise TypeError("checkpoint generator must implement state_dict")
        payload["task_generator"] = state_dict()
    if phase_state is not None:
        payload["phase_state"] = phase_state
    if provenance is not None:
        payload["provenance"] = dict(provenance)
    if torch.cuda.is_available(): payload["rng_cuda"] = torch.cuda.get_rng_state_all()
    periodic = save_atomic(payload, run_dir / "checkpoints", f"periodic-{step:09d}.ckpt")
    publish_latest(periodic, run_dir / "checkpoints")
    prune_periodic(run_dir / "checkpoints", int(config.get("checkpoint", {}).get("retain_periodic", 24)))
    return periodic


def _write_crash_bundle(run_dir: Path, events: EventWriter, *, step: int, phase: str,
                        config: dict, error: BaseException) -> Path:
    """Persist observed diagnostics without attempting a risky recovery.

    Crash reporting must itself be best effort: if a storage or CUDA failure
    caused the original error, inability to collect an optional field cannot
    hide that original traceback.
    """
    report_dir = run_dir / "crash_reports"
    report_dir.mkdir(exist_ok=True)
    cuda: dict[str, object] = {"available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        try:
            cuda.update(device=torch.cuda.get_device_name(0), allocated_bytes=torch.cuda.memory_allocated(0),
                        reserved_bytes=torch.cuda.memory_reserved(0), max_allocated_bytes=torch.cuda.max_memory_allocated(0))
        except RuntimeError as exc:
            cuda["diagnostic_error"] = repr(exc)
    latest = run_dir / "checkpoints" / "latest.ckpt"
    payload = {
        "created_unix": time.time(), "global_step": step, "phase": phase,
        "error": repr(error), "traceback": traceback.format_exc(), "recent_events": list(events.recent),
        "telemetry": telemetry_snapshot(), "cuda": cuda,
        "disk": {"free_bytes": shutil.disk_usage(run_dir).free, "total_bytes": shutil.disk_usage(run_dir).total},
        "latest_checkpoint": str(latest) if latest.is_file() else None,
        "config": config,
    }
    target = report_dir / f"crash-{int(time.time())}.json"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temporary, target)
    return target


def _phase_episode_factory(generator: object, phase: str, semantic_contains: object | None = None):
    """Bind every curriculum phase to a concrete real-task family.

    This is intentionally a small registry rather than a string decoration in
    TOML.  An unknown full phase is rejected before it can silently become the
    generic PPO loop.
    """
    def next_episode() -> tuple[object, tuple[int, ...]]:
        if phase == "rl_multihop_distractor":
            generator.min_hops, generator.max_hops = (4, 6)
            return generator.next_distractor()
        if phase == "rl_robustness":
            generator.min_hops, generator.max_hops = (4, 6)
            return generator.next_robustness()
        if phase == "rl_intersection":
            return generator.next_intersection(), ()
        if phase == "rl_semantic_hybrid":
            if semantic_contains is None:
                raise RuntimeError("semantic hybrid phase has no aligned semantic entity IDs")
            generator.min_hops, generator.max_hops = (2, 4)
            for _ in range(10_000):
                query, proof = generator.next()
                if semantic_contains(query.source):
                    # The family tag makes traces/evaluation disclose a real
                    # semantic attempt; it does not imply ANN success or proof.
                    from dataclasses import replace
                    return replace(query, family="semantic_hybrid"), proof
            raise RuntimeError("could not generate semantic task with an indexed source")
        if phase in {"rl_2_3_hop", "jetson_specialization", "behavior_cloning", "graph_ppo"}:
            generator.min_hops, generator.max_hops = (2, 3)
            return generator.next()
        raise RuntimeError(f"phase {phase!r} has no episode factory")
    return next_episode


def _semantic_search_callback(embeddings: object, backend: object):
    """Return a checked entity-ID -> CUDA exact search adapter.

    The source must actually have an aligned vector.  Missing coverage is an
    observable unavailable ANN action, never a fabricated vector.
    """
    ids = np.asarray(embeddings.entity_ids, dtype=np.uint32)
    vectors = embeddings.vectors
    def search(source: int, k: int) -> object:
        row = int(np.searchsorted(ids, np.uint32(source)))
        if row >= ids.size or int(ids[row]) != source:
            raise RuntimeError(f"semantic vector is unavailable for graph entity {source}")
        return embeddings.search(np.asarray(vectors[row], dtype=np.float32), k, backend)
    return search


def _measured_cuda_probe(cuda_backend: object, embeddings: object, graph: object, cuda_session: object, rows_requested: int = 4096) -> dict[str, float]:
    """One bounded, real CUDA score + real CSR timing observation."""
    start = time.perf_counter()
    rows = min(rows_requested, int(embeddings.vectors.shape[0]))
    query = np.asarray(embeddings.vectors[0], dtype=np.float32)
    scores = cuda_backend.scores(embeddings.vectors[:rows], query)
    score_ms = (time.perf_counter() - start) * 1_000.0
    source = int(embeddings.entity_ids[0])
    relation = int(graph.forward_relations[int(graph.forward_offsets[source])]) if graph.forward_offsets[source + 1] > graph.forward_offsets[source] else None
    start = time.perf_counter()
    expanded = cuda_session.expand([source], relation)
    expand_ms = (time.perf_counter() - start) * 1_000.0
    if scores.size != rows or not np.isfinite(scores).all():
        raise FloatingPointError("CUDA microbenchmark produced invalid scores")
    return {"cuda_score_rows": float(rows), "cuda_score_latency_ms": score_ms,
            "cuda_expand_candidates": float(len(expanded)), "cuda_expand_latency_ms": expand_ms}


def _latency_model_status(model: object) -> str:
    """Accept a latency surrogate only when it is directionally safe.

    A model trained on an under-varied probe set can fit noise with a negative
    candidate-count coefficient.  That would reward larger expansions as
    *faster*, which is the opposite of the low-latency objective.  Do not use
    such a surrogate for policy rewards; retain the explicit instruction cost
    until a representative measured model is available.
    """
    metrics = getattr(model, "metrics", {})
    try:
        mape = float(metrics.get("mape_percent", float("inf")))
    except (AttributeError, TypeError, ValueError):
        return "rejected_missing_validation"
    if not np.isfinite(mape) or mape > 50.0:
        return "rejected_validation_error"
    try:
        predictions = [float(model.predict("graph_expand", frontier_size=1, candidate_count=count))
                       for count in (1, 4, 16, 64, 256, 1024, 4096)]
    except (AttributeError, TypeError, ValueError):
        return "rejected_invalid_predictor"
    if not all(np.isfinite(value) and value >= 0.0 for value in predictions):
        return "rejected_invalid_prediction"
    if any(later + 1e-9 < earlier for earlier, later in zip(predictions, predictions[1:])):
        return "rejected_nonmonotonic_candidate_cost"
    return "accepted"


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _learned_evaluation_result(model: NavigatorPolicy, graph: object, query: object, proof: tuple[int, ...] | None,
                               device: torch.device, cuda_session: object, max_cuda_expand_edges: int,
                               semantic_search: object | None, navigator_candidate_cap: int):
    """Execute the checkpointed strategist on one held-out query.

    This evaluator intentionally receives no oracle action/path.  The stored
    proof is retained solely as an exported comparison artefact; correctness
    comes from ``GraphSearchEnv``'s independent validator.
    """
    from neuroseek.evaluation.baselines import BaselineResult
    from neuroseek.search.environment import GraphSearchEnv
    started = perf_counter()
    env = GraphSearchEnv(graph, query, (), cuda_session=cuda_session,
                         max_cuda_expand_edges=max_cuda_expand_edges,
                         candidate_ranker=_navigator_ranker(model, graph, device, navigator_candidate_cap),
                         semantic_search=semantic_search)
    result = None
    model.eval()
    # A bounded evaluator prevents an invalid policy from turning the final
    # phase into an unbounded search.  STOP is issued at the cap if required.
    with torch.no_grad():
        for _ in range(min(32, max(2, query.budget))):
            logits, _value = model(torch.as_tensor(env.observation(), device=device).unsqueeze(0))
            result = env.step(int(torch.argmax(logits[0]).item()))
            if result.done:
                break
        if result is not None and not result.done:
            result = env.step(OP_NAMES.index("STOP"))
    if result is None:
        raise RuntimeError("learned evaluation produced no instruction")
    return BaselineResult("neuroseek_learned", query.task_id, True,
                          env.answer, result.answer_correct, result.valid_proof,
                          result.nodes_visited, result.edges_examined, len(result.trace),
                          result.ann_calls, result.credits,
                          (perf_counter() - started) * 1_000.0,
                          tuple(env.proof_path), None), result.trace, int(result.navigator_ranked_candidates)


def _run_final_evaluation(model: NavigatorPolicy, graph: object, task_rows: list[tuple[object, tuple[int, ...] | None]],
                          device: torch.device, cuda_session: object, max_cuda_expand_edges: int,
                          embeddings: object, ann_backend: object, semantic_search: object | None,
                          run_dir: Path, provenance: dict[str, str], navigator_candidate_cap: int = 256,
                          evaluated_checkpoint: dict[str, str] | None = None) -> dict[str, float]:
    """Run once, export actual held-out comparisons, and never regenerate tasks."""
    from neuroseek.evaluation.baselines import (BaselineResult, ann_only_search, bfs_search, evaluate,
                                                fixed_relation_search, heuristic_hybrid_search)
    tasks = [query for query, _proof in task_rows]
    reports = {
        "bfs": evaluate(graph, tasks, bfs_search),
        "fixed_relation": evaluate(graph, tasks, fixed_relation_search),
        "hybrid": evaluate(graph, tasks, heuristic_hybrid_search),
    }
    ids = np.asarray(embeddings.entity_ids, dtype=np.uint32)
    vectors = embeddings.vectors
    def ann(query: object):
        row = int(np.searchsorted(ids, np.uint32(query.source)))
        if row >= ids.size or int(ids[row]) != query.source:
            return BaselineResult("ann_only", query.task_id, False, None, False, False, 0, 0, 0, 0, 0, 0.0,
                                  (), "source has no aligned semantic vector")
        return ann_only_search(embeddings, ann_backend, np.asarray(vectors[row], dtype=np.float32), query,
                               k=min(32, int(ids.size)))
    reports["ann_only"] = evaluate(graph, tasks, lambda _graph, query: ann(query))
    learned_rows = []
    traces: list[dict[str, object]] = []
    navigator_ranked_total = 0
    for query, proof in task_rows:
        row, trace, ranked_count = _learned_evaluation_result(model, graph, query, proof, device, cuda_session,
                                                              max_cuda_expand_edges, semantic_search, navigator_candidate_cap)
        navigator_ranked_total += ranked_count
        learned_rows.append(row)
        traces.append({"task": query.to_dict(), "expected_proof": list(proof) if proof else None,
                       "executed_trace": trace, "result": row.to_dict()})
    reports["neuroseek_learned"] = evaluate(graph, tasks, lambda _graph, query: next(item for item in learned_rows if item.task_id == query.task_id))
    exports = run_dir / "exports"
    serializable = {name: report.to_dict(False) for name, report in reports.items()}
    _write_json_atomic(exports / "final_metrics.json", {"provenance": provenance, "reports": serializable,
                                                          "learned_navigator_ranked_candidates": navigator_ranked_total})
    _write_json_atomic(exports / "reference_query_traces.json", traces[:32])
    _write_json_atomic(exports / "proof_examples.json", [item for item in traces if item["result"]["valid_proof"]][:32])
    # The trace-based distribution is recorded without invented aggregate
    # values; count only actual op prefixes emitted by the evaluator.
    distribution = {name: 0 for name in OP_NAMES}
    for trace in traces:
        for token in trace["executed_trace"]:
            op = str(token).split("(", 1)[0]
            if op in distribution:
                distribution[op] += 1
    _write_json_atomic(exports / "operator_distribution.json", distribution)
    with (exports / ".benchmark_comparison.csv.tmp").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["baseline", "task_count", "applicable_count", "answer_accuracy", "valid_proof_rate", "mean_nodes_visited", "mean_edges_examined", "mean_search_steps", "mean_ann_calls", "mean_compute_credits", "mean_latency_ms", "p95_latency_ms"])
        writer.writeheader()
        for name, report in reports.items():
            writer.writerow({"baseline": name, **{key: value for key, value in report.to_dict(False).items() if key != "results"}})
    os.replace(exports / ".benchmark_comparison.csv.tmp", exports / "benchmark_comparison.csv")
    _write_json_atomic(exports / "strategy_evolution.json", {"stage": "final", "reference_traces": traces[:32]})
    _write_json_atomic(exports / "hardware_summary.json", {"telemetry": telemetry_snapshot(), "provenance": provenance})
    _write_json_atomic(exports / "phase_summary.json", {"final_evaluation": serializable})
    if evaluated_checkpoint is None:
        raise RuntimeError("final evaluation requires an atomic evaluated checkpoint identity")
    _write_json_atomic(exports / "final_model_manifest.json", {"tag": "jetson-specialized", "provenance": provenance,
                                                                  "checkpoint": evaluated_checkpoint, "ann_backend": "cuda_exact"})
    # The durable JSONL remains the source of the curve; export a small CSV
    # projection for presentation tooling after the trainer flushes events.
    curve = exports / "training_curve.csv"
    with curve.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ts_unix", "global_step", "phase", "reward", "success_rate", "proof_validity", "nodes_per_query", "edges_per_query"])
        writer.writeheader()
        metrics_path = run_dir / "metrics.jsonl"
        if metrics_path.is_file():
            for line in metrics_path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                if row.get("category") == "TrainingEvent":
                    writer.writerow({key: row.get(key) for key in writer.fieldnames})
    learned = reports["neuroseek_learned"]
    return {"evaluation_task_count": float(len(tasks)), "learned_accuracy": float(learned.answer_accuracy or 0.0),
            "learned_proof_validity": float(learned.valid_proof_rate or 0.0),
            "learned_navigator_ranked_candidates": float(navigator_ranked_total),
            "bfs_accuracy": float(reports["bfs"].answer_accuracy or 0.0),
            "fixed_relation_accuracy": float(reports["fixed_relation"].answer_accuracy or 0.0),
            "hybrid_accuracy": float(reports["hybrid"].answer_accuracy or 0.0),
            "ann_only_accuracy": float(reports["ann_only"].answer_accuracy or 0.0)}


def _capture_reference_traces(model: NavigatorPolicy, graph: object, reference_rows: list[tuple[object, tuple[int, ...] | None]],
                              device: torch.device, cuda_session: object, max_cuda_expand_edges: int,
                              semantic_search: object | None, navigator_candidate_cap: int, run_dir: Path,
                              *, phase: str, global_step: int, elapsed_seconds: float) -> Path:
    """Persist actual programs for fixed held-out reference queries.

    The saved proof is presentation/audit context only; the evaluator executes
    the checkpointed policy without it.  One immutable file per entered phase
    lets a later UI compare the same query across training stages honestly.
    """
    rows: list[dict[str, object]] = []
    for query, proof in reference_rows:
        result, trace, ranked = _learned_evaluation_result(model, graph, query, proof, device, cuda_session,
                                                           max_cuda_expand_edges, semantic_search, navigator_candidate_cap)
        rows.append({"task": query.to_dict(), "expected_proof": list(proof) if proof else None,
                     "executed_trace": trace, "navigator_ranked_candidates": ranked,
                     "result": result.to_dict()})
    traces_dir = run_dir / "traces"
    target = traces_dir / f"reference-{global_step:09d}-{phase}.json"
    _write_json_atomic(target, {"phase": phase, "global_step": global_step,
                                "elapsed_seconds": elapsed_seconds, "traces": rows})
    index_path = traces_dir / "strategy_evolution.json"
    existing: list[dict[str, object]] = []
    if index_path.is_file():
        try:
            loaded = json.loads(index_path.read_text())
            if isinstance(loaded, list):
                existing = loaded
        except (OSError, json.JSONDecodeError):
            # A malformed prior index is a run-integrity error rather than a
            # reason to silently discard older strategy evidence.
            raise RuntimeError(f"strategy evolution index is malformed: {index_path}")
    existing.append({"phase": phase, "global_step": global_step, "elapsed_seconds": elapsed_seconds,
                     "file": target.name, "reference_count": len(rows)})
    _write_json_atomic(index_path, existing)
    return target


def main(argv: list[str] | None = None) -> int:
    # Tests and embedded launchers may invoke main more than once in a process.
    # A prior SIGTERM must not turn the next explicit launch into a no-op.
    global STOP
    STOP = False
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--parent-checkpoint", type=Path,
                        help="explicit checkpoint from an immutable-input-compatible parent run")
    args = parser.parse_args(argv)
    config = _read_toml(args.config)
    mode = config.get("run", {}).get("mode", "unknown")
    if mode != "smoke" and not (Path("data/processed/manifest.json").is_file()):
        raise RuntimeError("production/trial requires data/processed/manifest.json; run preprocessing first")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if mode != "smoke" and device.type != "cuda":
        raise RuntimeError("CUDA is mandatory outside explicitly named smoke mode")
    _seed(int(config.get("run", {}).get("seed", 1337)))
    graph = generator = cuda_session = semantic_search = semantic_backend = embeddings = cuda_backend = semantic_contains = None
    reference_rows: list[tuple[object, tuple[int, ...] | None]] = []
    max_cuda_expand_edges = int(config.get("training", {}).get("max_cuda_expand_edges", 262_144))
    navigator_candidate_cap = int(config.get("training", {}).get("navigator_candidate_cap", 256))
    if mode != "smoke":
        # Verify every immutable binary before opening mmap; existence alone is
        # not an integrity guarantee after an interrupted copy or disk fault.
        import subprocess
        checked = subprocess.run([sys.executable, "scripts/verify_data.py", "data/processed"], text=True, capture_output=True)
        if checked.returncode:
            raise RuntimeError(f"processed graph integrity check failed: {checked.stderr.strip()}")
        from neuroseek.data.graph import GraphMmap
        from neuroseek.data.tasks import TaskGenerator, load_task_jsonl
        from neuroseek.cuda_backend import CudaExactBackend
        from neuroseek.semantic import AlignedEmbeddings, CudaExactAnnBackend
        graph = GraphMmap("data/processed")
        split_manifest = json.loads(Path("data/processed/task_splits/manifest.json").read_text())
        heldout_ids: set[str] = set()
        for split_name in ("validation", "test"):
            filename = split_manifest.get(split_name)
            if not isinstance(filename, str):
                raise RuntimeError(f"held-out split manifest lacks {split_name} filename")
            path = Path("data/processed/task_splits") / filename
            if not path.is_file():
                raise RuntimeError(f"immutable held-out task artifact is missing: {path}")
            loaded_rows = load_task_jsonl(path)
            heldout_ids.update(query.task_id for query, _proof in loaded_rows)
            if split_name == "validation":
                seen_families: set[str] = set()
                for query, proof in loaded_rows:
                    if query.family not in seen_families:
                        reference_rows.append((query, proof))
                        seen_families.add(query.family)
                if not reference_rows:
                    raise RuntimeError("validation reference task artifact is empty")
        generator = TaskGenerator(graph, "train", int(config.get("run", {}).get("seed", 1337)), min_hops=2, max_hops=3,
                                  forbidden_task_ids=heldout_ids)
        # Full CSR is uploaded once.  Any native failure aborts trial/full
        # mode; no CPU fallback is permitted for the production graph path.
        cuda_backend = CudaExactBackend()
        cuda_backend.self_test()
        cuda_session = cuda_backend.create_graph_session(graph)
        semantic_root = Path("data/processed") / ("semantic_full" if mode == "full" else "semantic_bounded")
        embeddings = AlignedEmbeddings(semantic_root, expected_graph_entities=graph.manifest.entity_count,
                                       allow_partial=mode != "full")
        semantic_backend = CudaExactAnnBackend()
        semantic_search = _semantic_search_callback(embeddings, semantic_backend)
        aligned_ids = np.asarray(embeddings.entity_ids, dtype=np.uint32)
        def semantic_contains(entity: int) -> bool:
            row = int(np.searchsorted(aligned_ids, np.uint32(entity)))
            return row < aligned_ids.size and int(aligned_ids[row]) == entity
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "config.toml").write_bytes(args.config.read_bytes())
    manifest = args.run_dir / "manifest.json"
    provenance = {"config_sha256": _sha256_file(args.config)}
    if mode != "smoke":
        graph_manifest = Path("data/processed/manifest.json")
        split_manifest = Path("data/processed/task_splits/manifest.json")
        semantic_manifest = Path("data/processed") / ("semantic_full" if mode == "full" else "semantic_bounded") / "semantic_manifest.json"
        for required in (graph_manifest, split_manifest, semantic_manifest):
            if not required.is_file():
                raise RuntimeError(f"run provenance input is absent: {required}")
        provenance.update(graph_manifest_sha256=_sha256_file(graph_manifest),
                          task_split_manifest_sha256=_sha256_file(split_manifest),
                          semantic_manifest_sha256=_sha256_file(semantic_manifest))
    parent_start = load_checkpoint(args.parent_checkpoint, device) if args.parent_checkpoint is not None else None
    if args.parent_checkpoint is not None and args.resume:
        raise RuntimeError("parent checkpoint migration and same-run resume are mutually exclusive")
    if parent_start is not None:
        parent_provenance = parent_start.get("provenance")
        immutable_keys = ("graph_manifest_sha256", "task_split_manifest_sha256", "semantic_manifest_sha256", "mode")
        expected_parent = {**{key: provenance[key] for key in provenance if key != "config_sha256"}, "mode": str(mode)}
        if not isinstance(parent_provenance, dict) or any(parent_provenance.get(key) != expected_parent.get(key) for key in immutable_keys):
            raise RuntimeError("parent checkpoint immutable inputs differ; refusing accelerated migration")
    if not manifest.exists():
        dataset_manifest = json.loads(Path("data/processed/manifest.json").read_text()) if mode != "smoke" else None
        manifest_payload = {"run_id": args.run_dir.name, "mode": mode, "created_at": time.time(), "device": str(device), "torch": torch.__version__, "dataset_manifest": dataset_manifest, "provenance": provenance}
        if parent_start is not None:
            manifest_payload["derived_from"] = {"checkpoint": str(args.parent_checkpoint), "global_step": int(parent_start["global_step"]),
                                                    "elapsed_seconds": float(parent_start["elapsed_seconds"]),
                                                    "parent_provenance": parent_start["provenance"]}
        manifest.write_text(json.dumps(manifest_payload, indent=2) + "\n")
    else:
        existing_manifest = json.loads(manifest.read_text())
        if existing_manifest.get("completed_at"):
            raise RuntimeError("run has completed its configured wall-clock budget; create an explicit --new-run")
        if existing_manifest.get("mode") != mode:
            raise RuntimeError("run manifest mode differs from requested configuration; refusing incompatible resume")
        # Smoke is a deliberately bounded fault-injection harness.  Its tests
        # resume with a shortened *wall-clock limit* after SIGTERM, so it is
        # intentionally exempt from production input binding.  Trial/full
        # checkpoints remain fail-closed on every immutable input below.
        if mode != "smoke" and existing_manifest.get("provenance") != provenance:
            raise RuntimeError("run input provenance differs (config/graph/semantic/split); refusing incompatible checkpoint resume; create --new-run")
    checkpoint_provenance = {**provenance, "mode": str(mode), "run_id": args.run_dir.name}
    events = EventWriter(args.run_dir)
    thermal_policy = ThermalSafetyPolicy.from_config(config)
    thermal_collector = None
    if thermal_policy is not None:
        thermal_collector = JetsonTelemetryCollector(
            tegrastats_interval_seconds=float(config["telemetry"].get("tegrastats_interval_seconds", 10.0))
        )
    last_thermal_level = ""
    model = NavigatorPolicy().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get("training", {}).get("lr", 3e-4)))
    checkpoint_dir = args.run_dir / "checkpoints"
    start = load_latest(checkpoint_dir, device,
                        expected_provenance=checkpoint_provenance if mode != "smoke" else None) if args.resume else parent_start
    if args.resume and start is None and has_checkpoint_candidates(checkpoint_dir):
        raise RuntimeError("resume requested but no valid checkpoint exists; refusing to restart this run at step zero")
    step, best, elapsed_before = 0, float("-inf"), 0.0
    phase = "behavior_cloning"
    restored_phase_state: dict | None = None
    if start:
        model.load_state_dict(start["model"]); optimizer.load_state_dict(start["optimizer"]); step = int(start["global_step"]); best = float(start["best_metric"]); phase = str(start["phase"])
        elapsed_before = float(start.get("elapsed_seconds", 0.0))
        # Resume the stochastic stream when the checkpoint contains it. Older
        # checkpoints remain loadable, but are explicitly not bit-identical.
        if "rng_python" in start: random.setstate(start["rng_python"])
        if "rng_numpy" in start: np.random.set_state(start["rng_numpy"])
        if "rng_torch" in start: torch.set_rng_state(start["rng_torch"])
        if torch.cuda.is_available() and "rng_cuda" in start: torch.cuda.set_rng_state_all(start["rng_cuda"])
        if generator is not None:
            if "task_generator" not in start:
                raise RuntimeError("resume checkpoint lacks task generator state; refusing task-stream drift")
            exclusions_migrated = generator.load_state_dict(start["task_generator"], allow_additional_forbidden=True)
        else:
            exclusions_migrated = False
        if "phase_state" in start:
            if not isinstance(start["phase_state"], dict):
                raise RuntimeError("checkpoint phase state is malformed")
            restored_phase_state = start["phase_state"]
        events.emit("RecoveredFromCheckpoint", global_step=step, phase=phase,
                    checkpoint=str(start.get("_checkpoint_source", "unknown")),
                    elapsed_seconds=elapsed_before, heldout_exclusions_migrated=exclusions_migrated)
        if parent_start is not None:
            events.emit("DerivedAcceleratedRun", global_step=step, phase=phase,
                        parent_checkpoint=str(args.parent_checkpoint), elapsed_seconds=elapsed_before)
    signal.signal(signal.SIGTERM, _stop); signal.signal(signal.SIGINT, _stop)
    budget = float(config.get("run", {}).get("budget_seconds", 60))
    _validate_phase_schedule(config, mode, budget)
    started_monotonic = time.monotonic()
    deadline = started_monotonic + max(0.0, budget - elapsed_before)
    batch = int(config.get("training", {}).get("batch_size", 16))
    ppo_entropy_coef = float(config.get("training", {}).get("ppo_entropy_coef", 0.01))
    ppo_action_temperature = float(config.get("training", {}).get("ppo_action_temperature", 1.0))
    rl_bc_anchor_interval = int(config.get("training", {}).get("rl_bc_anchor_interval", 0))
    latency_penalty_coefficient = float(config.get("training", {}).get("latency_penalty_coefficient", 0.01))
    instruction_penalty = float(config.get("training", {}).get("instruction_penalty", 0.0))
    if ppo_entropy_coef < 0.0 or not np.isfinite(ppo_entropy_coef):
        raise ValueError("training.ppo_entropy_coef must be finite and non-negative")
    if ppo_action_temperature <= 0.0 or not np.isfinite(ppo_action_temperature):
        raise ValueError("training.ppo_action_temperature must be finite and positive")
    if rl_bc_anchor_interval < 0:
        raise ValueError("training.rl_bc_anchor_interval must be non-negative")
    if latency_penalty_coefficient < 0.0 or not np.isfinite(latency_penalty_coefficient):
        raise ValueError("training.latency_penalty_coefficient must be finite and non-negative")
    if instruction_penalty < 0.0 or not np.isfinite(instruction_penalty):
        raise ValueError("training.instruction_penalty must be finite and non-negative")
    checkpoint_every = float(config.get("checkpoint", {}).get("seconds", 30))
    next_checkpoint = time.monotonic() + checkpoint_every
    minimum_free_disk_bytes = _minimum_free_disk_bytes(config, mode)
    next_disk_check = time.monotonic()
    halt_reason: str | None = None
    phase_state: dict[str, object] = restored_phase_state or {"entered": [], "hardware_samples": [], "hardware_records": []}
    phase_state.setdefault("entered", [])
    phase_state.setdefault("hardware_samples", [])
    phase_state.setdefault("hardware_records", [])
    # Raw hardware observations belong in the durable event log, not in every
    # model checkpoint.  A bounded suffix is sufficient for the small online
    # regressor and keeps checkpoint size/recovery time stable over 50 hours.
    hardware_window = int(config.get("hardware", {}).get("checkpoint_window", 256))
    hardware_sample_interval = float(config.get("hardware", {}).get("sample_interval_seconds", 1.0))
    if hardware_window < 8:
        raise ValueError("hardware.checkpoint_window must be at least 8")
    if not np.isfinite(hardware_sample_interval) or hardware_sample_interval <= 0.0:
        raise ValueError("hardware.sample_interval_seconds must be finite and positive")
    for key in ("hardware_samples", "hardware_records"):
        if not isinstance(phase_state[key], list):
            raise RuntimeError(f"checkpoint {key} is malformed")
        phase_state[key] = phase_state[key][-hardware_window:]
    if not isinstance(phase_state["entered"], list) or not all(isinstance(item, str) for item in phase_state["entered"]):
        raise RuntimeError("checkpoint entered phase state is malformed")
    finish_after_evaluation = False
    try:
        while not STOP and time.monotonic() < deadline:
            elapsed_total = elapsed_before + (time.monotonic() - started_monotonic)
            scheduled_phase = _phase_for_elapsed(config, elapsed_total, budget)
            if minimum_free_disk_bytes and time.monotonic() >= next_disk_check:
                # Do not wait for the filesystem to be completely full: a
                # checkpoint and crash report still need writable space.  The
                # 60-second cadence is deliberately independent of minibatch
                # rate so it cannot become a training throughput tax.
                next_disk_check = time.monotonic() + 60.0
                free_disk = shutil.disk_usage(args.run_dir).free
                if free_disk < minimum_free_disk_bytes:
                    path = _checkpoint(model, optimizer, args.run_dir, phase, step, best, config, elapsed_total,
                                       generator=generator, phase_state=phase_state, provenance=checkpoint_provenance)
                    events.emit("WarningEvent", global_step=step, phase=phase,
                                warning="disk_reserve_exhausted", requested_action="checkpoint_and_stop",
                                free_disk_bytes=free_disk, minimum_free_disk_bytes=minimum_free_disk_bytes,
                                checkpoint=str(path.name))
                    halt_reason = "disk_reserve_exhausted"
                    break
            if graph is not None and scheduled_phase not in phase_state["entered"]:
                trace_path = _capture_reference_traces(model, graph, reference_rows, device, cuda_session,
                                                      max_cuda_expand_edges, semantic_search, navigator_candidate_cap,
                                                      args.run_dir, phase=scheduled_phase, global_step=step,
                                                      elapsed_seconds=elapsed_total)
                phase_state["entered"].append(scheduled_phase)
                events.emit("CurriculumEvent", phase=scheduled_phase, global_step=step,
                            elapsed_seconds=elapsed_total, reference_trace=str(trace_path.name),
                            reference_query_count=len(reference_rows))
            if scheduled_phase in {"cuda_search_microbenchmarks", "hardware_cost_model"}:
                next_probe = float(phase_state.get("next_hardware_probe_elapsed", 0.0))
                if elapsed_total < next_probe:
                    # Benchmark phases collect independent real observations;
                    # spinning faster only creates duplicate fsync-heavy JSONL
                    # rows and distorts the storage/thermal state being modeled.
                    time.sleep(min(0.1, next_probe - elapsed_total))
                    continue
                phase_state["next_hardware_probe_elapsed"] = elapsed_total + hardware_sample_interval
            if graph is None:
                # Explicitly named smoke-only tensor exercise.
                states = _state(device, batch, 32); logits, values = model(states); dist = torch.distributions.Categorical(logits=logits / ppo_action_temperature); actions = dist.sample(); old_logprob = dist.log_prob(actions).detach(); reward = (actions == 2).float(); returns = reward + 0.99 * values.detach(); stats = ppo_update(model, optimizer, states, actions, old_logprob, returns, reward, 0.2, ppo_entropy_coef, 0.5, action_temperature=ppo_action_temperature); observed = {"reward": float(reward.mean()), "success_rate": 0.0, "proof_validity": 0.0, "nodes_per_query": 0.0, "edges_per_query": 0.0, "credits_per_query": 0.0, "operator_sample": OP_NAMES[int(actions[0])]}
            elif scheduled_phase == "cuda_search_microbenchmarks":
                phase = scheduled_phase
                if cuda_backend is None or embeddings is None or cuda_session is None:
                    raise RuntimeError("CUDA benchmark phase has no initialized production backend")
                observed = _measured_cuda_probe(cuda_backend, embeddings, graph, cuda_session)
                observed.update(reward=0.0, success_rate=0.0, proof_validity=0.0, nodes_per_query=0.0,
                                edges_per_query=0.0, credits_per_query=0.0, operator_sample="CUDA_MICROBENCH")
                phase_state["hardware_samples"].append(observed)
                phase_state["hardware_samples"] = phase_state["hardware_samples"][-hardware_window:]
                events.emit("HardwareEvent", phase=phase, **observed)
            elif scheduled_phase == "hardware_cost_model":
                phase = scheduled_phase
                if cuda_backend is None or embeddings is None or cuda_session is None:
                    raise RuntimeError("hardware cost-model phase has no initialized production backend")
                row_choices = (1024, 2048, 4096, 8192, 16384)
                observed = _measured_cuda_probe(cuda_backend, embeddings, graph, cuda_session,
                                                row_choices[len(phase_state["hardware_records"]) % len(row_choices)])
                samples = phase_state["hardware_samples"]
                samples.append(observed)
                records = phase_state["hardware_records"]
                records.extend([
                    {"operation": "exact_scores", "latency_ms": observed["cuda_score_latency_ms"],
                     "rows": observed["cuda_score_rows"], "dims": float(embeddings.manifest.dimension)},
                    {"operation": "graph_expand", "latency_ms": observed["cuda_expand_latency_ms"],
                     "frontier_size": 1.0, "candidate_count": observed["cuda_expand_candidates"]},
                ])
                phase_state["hardware_samples"] = phase_state["hardware_samples"][-hardware_window:]
                phase_state["hardware_records"] = phase_state["hardware_records"][-hardware_window:]
                if len(records) >= 6:
                    from neuroseek.cost_model.model import OperationRecord, save_model, train_cost_model
                    normalized = [OperationRecord(str(item["operation"]), float(item["latency_ms"]),
                                                  {key: float(value) for key, value in item.items() if key not in {"operation", "latency_ms"}},
                                                  source="trainer_hardware_phase", line=index + 1)
                                  for index, item in enumerate(records)]
                    cost_model = train_cost_model(normalized)
                    phase_state["cost_model"] = cost_model.to_dict()
                    save_model(cost_model, args.run_dir / "exports" / "hardware_cost_model.json", records=normalized)
                # The model is intentionally transparent until enough actual
                # observations exist; mean real latency is not presented as a
                # predictive regression score.
                observed.update(reward=0.0, success_rate=0.0, proof_validity=0.0, nodes_per_query=0.0,
                                edges_per_query=0.0, credits_per_query=0.0, operator_sample="HARDWARE_COST_SAMPLE",
                                hardware_sample_count=float(len(samples)),
                                observed_mean_cuda_latency_ms=float(np.mean([float(item["cuda_score_latency_ms"]) + float(item["cuda_expand_latency_ms"]) for item in samples])),
                                cost_model_trained=float("cost_model" in phase_state))
                events.emit("HardwareEvent", phase=phase, **observed)
            elif scheduled_phase == "deterministic_final_evaluation":
                phase = scheduled_phase
                from neuroseek.data.tasks import load_task_jsonl
                if phase_state.get("final_evaluation_complete"):
                    finish_after_evaluation = True
                    break
                if cuda_session is None or embeddings is None or semantic_backend is None:
                    raise RuntimeError("deterministic final evaluation requires initialized CUDA graph and ANN backends")
                split_manifest = json.loads(Path("data/processed/task_splits/manifest.json").read_text())
                filename = split_manifest.get("test")
                if not isinstance(filename, str):
                    raise RuntimeError("held-out split manifest lacks test filename")
                task_path = Path("data/processed/task_splits") / filename
                # Bind exports to the exact in-memory policy before any
                # evaluation work.  The periodic generation is retained too;
                # final.ckpt is an immutable named publication for the final
                # model manifest, never a fragile pointer to a later latest.
                pre_eval = _checkpoint(model, optimizer, args.run_dir, phase, step, best, config, elapsed_total,
                                       generator=generator, phase_state=phase_state, provenance=checkpoint_provenance)
                final_path = save_atomic(torch.load(pre_eval, map_location="cpu"), checkpoint_dir, "final.ckpt")
                evaluated_checkpoint = {"filename": final_path.name, "sha256": _sha256_file(final_path)}
                evaluation_metrics = _run_final_evaluation(model, graph, load_task_jsonl(task_path), device, cuda_session,
                                                           max_cuda_expand_edges, embeddings, semantic_backend, semantic_search,
                                                           args.run_dir, provenance, navigator_candidate_cap,
                                                           evaluated_checkpoint)
                phase_state["final_evaluation_complete"] = True
                observed = {"reward": 0.0, "success_rate": evaluation_metrics["learned_accuracy"], "proof_validity": evaluation_metrics["learned_proof_validity"], "nodes_per_query": 0.0,
                            "edges_per_query": 0.0, "credits_per_query": 0.0, "operator_sample": "DETERMINISTIC_EVALUATION",
                            **evaluation_metrics}
                finish_after_evaluation = True
            elif scheduled_phase == "behavior_cloning":
                phase = scheduled_phase
                factory = _phase_episode_factory(generator, phase, semantic_contains)
                observed = _behavior_clone(model, optimizer, graph, generator, device, batch, cuda_session, max_cuda_expand_edges, navigator_candidate_cap, episode_factory=factory, semantic_search=semantic_search)
                observed["operator_sample"] = "DEMONSTRATION"
            else:
                phase = scheduled_phase
                factory = _phase_episode_factory(generator, phase, semantic_contains)
                predictor = None
                latency_model_status = "not_requested"
                if phase == "jetson_specialization":
                    from neuroseek.cost_model.model import CostModel
                    if "cost_model" not in phase_state:
                        raise RuntimeError("Jetson specialization requires an observed hardware cost model")
                    model_cost = CostModel.from_dict(phase_state["cost_model"])
                    latency_model_status = _latency_model_status(model_cost)
                    if latency_model_status == "accepted":
                        predictor = lambda result: model_cost.predict("graph_expand", frontier_size=1,
                                                                      candidate_count=result.nodes_visited,
                                                                      edge_count=result.edges_examined)
                states, actions, old_logprob, returns, graph_metrics = _collect_real_rollouts(model, graph, generator, device, batch, cuda_session, max_cuda_expand_edges, navigator_candidate_cap, episode_factory=factory, semantic_search=semantic_search, hardware_cost_predictor=predictor, action_temperature=ppo_action_temperature, latency_penalty_coefficient=latency_penalty_coefficient, instruction_penalty=instruction_penalty)
                advantages = returns - returns.mean()
                stats = ppo_update(model, optimizer, states, actions, old_logprob, returns, advantages, 0.2, ppo_entropy_coef, 0.5, action_temperature=ppo_action_temperature)
                observed = {**graph_metrics, "policy_loss": stats.policy_loss, "value_loss": stats.value_loss, "entropy": stats.entropy, "kl": stats.kl, "gradient_norm": stats.grad_norm, "operator_sample": OP_NAMES[int(actions[0])], "latency_model_status": latency_model_status}
                # PPO can otherwise reinforce a transient all-SEED failure
                # batch until entropy vanishes.  A configurable BC anchor
                # preserves executable proof-program priors while rollout
                # reward remains the primary update signal.
                anchor_count = int(phase_state.get("rl_anchor_count", 0)) + 1
                phase_state["rl_anchor_count"] = anchor_count
                if rl_bc_anchor_interval and anchor_count % rl_bc_anchor_interval == 0:
                    anchor = _behavior_clone(model, optimizer, graph, generator, device, batch, cuda_session,
                                             max_cuda_expand_edges, navigator_candidate_cap,
                                             episode_factory=factory, semantic_search=semantic_search)
                    observed["bc_anchor_policy_loss"] = anchor["policy_loss"]
                    observed["bc_anchor_navigator_loss"] = anchor["navigator_loss"]
            step += batch
            thermal_observed = thermal_collector.sample() if thermal_collector is not None else {}
            observed = {"global_step": step, "elapsed_seconds": elapsed_total, "phase": phase,
                        "cuda": device.type == "cuda", "cuda_graph_session": cuda_session is not None,
                        **telemetry_snapshot(), **thermal_observed, **observed}
            events.emit("TrainingEvent", **observed)
            if "live_search_trace" in observed:
                # The dashboard reads this durable record independently.  A
                # TUI crash or a terminal disconnect cannot lose the actual
                # search trace needed for later debugging/presentation.
                events.emit("SearchTraceEvent", global_step=step, phase=phase,
                            task_id=observed["live_search_task"], family=observed["live_search_family"],
                            trace=observed["live_search_trace"], result=observed["live_search_result"],
                            operator_distribution=observed["operator_distribution"])
            thermal_decision = thermal_policy.evaluate(thermal_observed) if thermal_policy is not None else None
            thermal_transition = thermal_decision is not None and thermal_decision.level != last_thermal_level
            if thermal_transition:
                last_thermal_level = thermal_decision.level
                if thermal_decision.level in {"warning", "critical"}:
                    events.emit("WarningEvent", global_step=step, phase=phase,
                                warning=thermal_decision.reason, thermal_level=thermal_decision.level,
                                requested_action=thermal_decision.action,
                                temperature_c=thermal_decision.temperature_c)
            if observed["reward"] > best:
                best = observed["reward"]
                checkpoint = _checkpoint(model, optimizer, args.run_dir, phase, step, best, config, elapsed_total, generator=generator, phase_state=phase_state, provenance=checkpoint_provenance)
                save_atomic(torch.load(checkpoint, map_location="cpu"), args.run_dir / "checkpoints", "best.ckpt")
                events.emit("NewBestPolicy", global_step=step, reward=best)
            if thermal_transition and thermal_decision is not None and thermal_decision.action in {"checkpoint", "checkpoint_stop", "pause"}:
                path = _checkpoint(model, optimizer, args.run_dir, phase, step, best, config, elapsed_total, generator=generator, phase_state=phase_state, provenance=checkpoint_provenance)
                events.emit("CheckpointEvent", global_step=step, checkpoint=str(path.name), reason="thermal_safety")
                if thermal_decision.action == "checkpoint_stop":
                    # A prolonged critical temperature can make a fixed
                    # 50-hour budget scientifically useless through severe
                    # throttling.  Preserve progress, then require an
                    # operator to correct cooling before explicit resume.
                    halt_reason = "critical_temperature"
                    break
                if thermal_decision.action == "pause":
                    # The pause is intentionally an application-level safe
                    # point.  It never changes Jetson thermal controls and
                    # checks SIGTERM/SIGINT between bounded polling intervals.
                    pause_seconds = float(config.get("telemetry", {}).get("pause_poll_seconds", 10.0))
                    while not STOP:
                        time.sleep(max(0.1, pause_seconds))
                        refreshed = thermal_collector.sample() if thermal_collector is not None else {}
                        resumed = thermal_policy.evaluate(refreshed) if thermal_policy is not None else None
                        events.emit("HardwareEvent", global_step=step, phase=phase, thermal_pause=True, **refreshed)
                        if resumed is None or resumed.level != "critical":
                            last_thermal_level = resumed.level if resumed is not None else ""
                            events.emit("WarningEvent", global_step=step, phase=phase, warning="thermal_pause_released")
                            break
            if time.monotonic() >= next_checkpoint:
                path = _checkpoint(model, optimizer, args.run_dir, phase, step, best, config, elapsed_total, generator=generator, phase_state=phase_state, provenance=checkpoint_provenance)
                events.emit("CheckpointEvent", global_step=step, checkpoint=str(path.name))
                next_checkpoint = time.monotonic() + checkpoint_every
            if finish_after_evaluation:
                break
        elapsed_total = elapsed_before + (time.monotonic() - started_monotonic)
        path = _checkpoint(model, optimizer, args.run_dir, phase, step, best, config, elapsed_total, generator=generator, phase_state=phase_state, provenance=checkpoint_provenance)
        events.emit("PhaseComplete", global_step=step, stopped=bool(STOP or halt_reason),
                    reason=halt_reason, checkpoint=str(path.name))
        if not STOP and halt_reason is None:
            updated = json.loads(manifest.read_text())
            updated["completed_at"] = time.time()
            temporary = manifest.with_suffix(".tmp")
            temporary.write_text(json.dumps(updated, indent=2) + "\n")
            os.replace(temporary, manifest)
        return 0
    except Exception as exc:
        report = _write_crash_bundle(args.run_dir, events, step=step, phase=phase, config=config, error=exc)
        events.emit("ErrorEvent", global_step=step, error=repr(exc), crash_report=str(report.name))
        raise
    finally:
        # Events are append+flush on every update for a responsive dashboard,
        # but their fsync is intentionally batched to avoid millions of block
        # device barriers across a 50-hour run.  Every normal/error/signal
        # exit reaches this safe point and forces the final suffix durable.
        events.sync()
        if cuda_session is not None:
            cuda_session.close()


if __name__ == "__main__":
    raise SystemExit(main())
