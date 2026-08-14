# NEUROSEEK final measurement record

This is the submission record for the completed Jetson low-latency model.  It
collects the measured outcomes, provenance, environment snapshot, and exact
artifact locations without treating planned work or synthetic smoke checks as
scientific results.

## Result at a glance

The final checkpoint was compared with its parent on the same fixed 256-task
held-out set, immutable Wikidata5M graph, semantic artifact, and `cuda_exact`
backend.

| Metric | Parent | Final | Change |
| --- | ---: | ---: | ---: |
| Mean end-to-end latency | 31.92 ms | 31.11 ms | -2.56% |
| p95 end-to-end latency | 57.10 ms | 53.02 ms | -7.14% |
| Mean compute credits/task | 233.98 | 146.02 | -37.60% |
| Answer accuracy / valid-proof rate | 98.05% | 97.27% | -0.78 points (2 tasks) |

The result is therefore an explicit efficiency/accuracy trade-off: the final
model lowers measured latency and search cost, while losing two correct tasks.
Every reported answer is checked by the independent Rust proof validator.

## Dataset and model identity

| Item | Measured value |
| --- | --- |
| Graph source | Wikidata5M-derived immutable CSR graph |
| Entities | 4,594,149 |
| Relations | 822 |
| Original triples | 20,614,279 |
| Held-out evaluation | 256 fixed tasks |
| Run | `latency-optimization-20260813T1255EDT` |
| Device/backend | CUDA / `cuda_exact` |
| Final checkpoint SHA-256 | `0c629cb5e94854b9dd8039fb457162a71d34b45057fe2591f6d3439f6875a174` |
| Config SHA-256 | `658104380fd26158f44586e6bea8c0fe6dc891db8c9fa0eb34c262a5cb0baaa8` |

The graph, semantic artifact, task-split, and configuration hashes are stored
with the run in [final_model_manifest.json](../runs/latency-optimization-20260813T1255EDT/exports/final_model_manifest.json).

## Baseline comparison

The complete fixed-test aggregates are in
[benchmark_comparison.csv](../runs/latency-optimization-20260813T1255EDT/exports/benchmark_comparison.csv).
The learned model is compared fairly only with its parent: the `hybrid` and
`fixed_relation` baselines receive a hand-written relation program, while the
learned model selects operations itself.

| Method | Applicable tasks | Accuracy | Valid proofs | Mean latency | p95 latency | Mean credits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BFS | 224 | 52.68% | 21.88% | 24.15 ms | 88.33 ms | 5,480.02 |
| Fixed relation | 224 | 99.11% | 99.11% | 5.28 ms | 28.92 ms | 236.28 |
| Hand-written hybrid | 256 | 99.22% | 99.22% | 4.66 ms | 28.02 ms | 207.84 |
| ANN only | 256 | 0.00% | 0.00% | 1,655.58 ms | 1,674.11 ms | 4,594,149.00 |
| Final learned NEUROSEEK | 256 | 97.27% | 97.27% | 31.11 ms | 53.02 ms | 146.02 |

## Task-family recheck

The post-completion recheck reran the immutable parent and final checkpoints
on the same 256 tasks.  It is a structural/cost breakdown, not a second
latency measurement.  Both changed failures were `path` tasks; the other four
families retained their parent accuracy.

| Family | Tasks | Parent accuracy | Final accuracy | Final minus parent credits |
| --- | ---: | ---: | ---: | ---: |
| path | 96 | 97.92% | 95.83% | -4.38 |
| distractor | 64 | 98.44% | 98.44% | -23.94 |
| intersection | 32 | 100.00% | 100.00% | 0.00 |
| semantic_hybrid | 32 | 96.88% | 96.88% | -642.72 |
| robustness | 32 | 96.88% | 96.88% | 0.00 |

See [task_family_recheck.json](../runs/latency-optimization-20260813T1255EDT/analysis/task_family_recheck.json)
and [task_level_recheck.csv](../runs/latency-optimization-20260813T1255EDT/analysis/task_level_recheck.csv)
for the source rows.

## Environment measurements

The final-run hardware sample recorded 7.44 GiB total RAM, 4.55 GiB RAM used,
and a 51.03 C GPU thermal sensor value.  The submission-time host probe on
2026-08-13 observed Linux `5.15.136-tegra` on `aarch64`, 7.4 GiB RAM, 84 GiB
free disk, and 3.0 GiB free swap.  These are snapshots, not a sustained power,
temperature, or throughput characterization.

The full final-run telemetry snapshot is
[hardware_summary.json](../runs/latency-optimization-20260813T1255EDT/exports/hardware_summary.json).

## Verification completed for this submission

On the submission host, the current source passed:

| Check | Result |
| --- | --- |
| `PYTHONPATH=python pytest -q` | 60 passed |
| `cargo test --workspace -q` | 13 passed |

These checks establish source-level behavior.  The published latency outcome
above comes solely from the retained CUDA final-evaluation artifacts, not from
these unit tests.

## Artifact index

- [Interactive final report](../reports/latency-final-presentation/report.html)
- [Release notes](../reports/latency-final-presentation/RELEASE_NOTES.md)
- [Final metrics](../runs/latency-optimization-20260813T1255EDT/exports/final_metrics.json)
- [Run manifest](../runs/latency-optimization-20260813T1255EDT/manifest.json)
- [Hardware summary](../runs/latency-optimization-20260813T1255EDT/exports/hardware_summary.json)
- [Proof examples](../runs/latency-optimization-20260813T1255EDT/exports/proof_examples.json)
- [Training curve](../runs/latency-optimization-20260813T1255EDT/exports/training_curve.csv)
- [Operator distribution](../runs/latency-optimization-20260813T1255EDT/exports/operator_distribution.json)

## Scope and limits

This is a single-Jetson, one-fixed-test-set evaluation.  It has no repeated-run
confidence intervals, external-SOTA comparison, natural-language relation
parsing evaluation, or end-to-end LLM quality measurement.  The exact CUDA
backend is correctness-first and uses a deterministic host-side top-k merge;
it should not be presented as a pure GPU ANN benchmark.
