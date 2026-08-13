# NEURO-ISA

NEURO-ISA is the bounded instruction language used by NEUROSEEK search
policies. `rust/core` is its deterministic CPU semantic reference. CUDA
kernels may accelerate equivalent bulk work, but CUDA output must be checked
against this reference and must preserve the same resource accounting and
proof evidence.

## State and invariants

The VM has the query outside the mutable machine state, frontier registers
`F0` through `F3`, answer candidates `A`, remaining credits `B`, a backtrack
history, and an observed proof. Frontiers hold compact `uint32` node IDs and
scores only. The graph hot path uses CSR `uint64` offsets plus adjacent
`uint32` targets and `uint16` relations; labels never enter the VM.

Every instruction is atomic. If it cannot run—for example, its deterministic
cost exceeds `B`, an ANN provider is absent, or a register argument is
invalid—the frontiers, proof, budget, depth, and successful-operation counters
are restored. Only `attempted_instructions` and `failed_instructions` record
the rejection. This prevents a scheduler from continuing with a partly
expanded frontier after an error.

| Instruction | Semantics | Deterministic credit model |
| --- | --- | --- |
| `SEED(node)` | Write a validated node to `F0`, score 1.0. | 1 |
| `ANN(k)` | Ask the configured ANN provider for at most `k` semantic candidates, normalize duplicate node IDs by maximum score, then write `F0`. | `k` |
| `EXPAND_REL(r)` | Traverse CSR neighbors of every `F0` node having relation `r`, write normalized targets to `F0`, and append actual traversed edges to the proof. | emitted edges |
| `EXPAND_ANY` | Same traversal without a relation predicate. | emitted edges |
| `FILTER(r)` | Keep `F0` nodes that themselves have at least one outgoing edge of relation `r`. | input frontier size |
| `INTERSECT(Fi)` | Keep node IDs shared by `F0` and `Fi`. | both operand sizes |
| `UNION(Fi)` | Union `F0` and `Fi`, taking the maximum duplicate score. | both operand sizes |
| `PRUNE(k)` / `TOPK(k)` | Sort score descending then node ID ascending, retaining the first `k`. | input frontier size |
| `VERIFY` | Copy `F0` to `A`; the first deterministic candidate becomes the proposed proof answer. | 1 |
| `BACKTRACK` | Restore the previous saved frontier-register snapshot. | 1 |
| `PREFETCH` / `EVICT` | Explicit cache-intent events. They have no hidden cache side effects in the CPU reference. | 1 |
| `STOP` | Close the program; later instructions are rejected. | 1 |

`EXPAND_*` records every CSR edge actually selected by that instruction, not a
task-generator demonstration path. `VERIFY` does not imply correctness. A
separate `validate_proof()` verifies that every required query edge was
observed, every proof edge is grounded in CSR, and the answer is an allowed
answer supported by evidence. This validator is independent of the policy and
of any teacher trajectory.

## Observability

For each accepted instruction the VM emits an in-memory `VmStep` containing:

- opcode, frontier size before/after, proposed answer, proof-edge count, and
  credits remaining;
- attempted/successful/failed instruction counts;
- nodes visited, edges examined, ANN calls, credits, estimated CSR bytes,
  maximum depth, backtracks, prefetches, evictions, frontier peak, and proof
  edge count.

The CLI serializes those values as JSONL. Wall time and physical GPU telemetry
are measured by the caller/trainer rather than fabricated by the VM.

## JSONL acceptance runner

`neuroseek-native` accepts a sequence of newline-delimited records. A valid
`graph` record must appear before each `program` record. The process remains
bounded by the program's explicit credits and produces `graph_loaded`, one
`vm_step` per accepted operation, then `vm_result`; malformed input and VM
errors become structured `*_error` lines rather than panics.

```bash
cargo run -p neuroseek-native -- --jsonl rust/cli/tests/fixtures/path_proof.jsonl
```

The checked fixture constructs the two-hop CSR `0 -7-> 1 -8-> 2`, executes
`SEED`, two relation expansions, `VERIFY`, and `STOP`, and reports
`"proof_valid":true` for answer `2`. It is intentionally small and is an
acceptance/diagnostic path, not the production Wikidata5M loader.

Input shape:

```json
{"type":"graph","nodes":3,"edges":[{"src":0,"relation":7,"dst":1}]}
{"type":"program","budget":16,"instructions":[{"op":"SEED","node":0},{"op":"STOP"}]}
```

Supported operation names are `SEED`, `ANN`, `EXPAND_REL`, `EXPAND_ANY`,
`FILTER`, `INTERSECT`, `UNION`, `PRUNE`, `TOP_K`, `VERIFY`, `BACKTRACK`,
`PREFETCH`, `EVICT`, and `STOP`. `ANN` deliberately returns an error in this
standalone acceptance runner because it has no configured semantic backend;
the production native boundary must supply one explicitly.
