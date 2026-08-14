# NEUROSEEK Jetson low-latency model

This package contains the completed latency-specialized NEUROSEEK checkpoint
and the immutable artifacts required to inspect its final held-out evaluation.

## Final held-out result

The derived checkpoint was evaluated on 256 fixed Wikidata5M-based test tasks
using the CUDA exact backend.  Against its parent checkpoint, it reduced mean
end-to-end latency from 31.92 ms to 31.11 ms (-2.56%), p95 latency from 57.10
ms to 53.02 ms (-7.14%), and mean compute credits from 233.98 to 146.02
(-37.59%).  Answer accuracy and independently validated proof rate changed
from 98.05% to 97.27% (-0.78 percentage points).

## Contents

- `checkpoints/final.ckpt`: immutable evaluated checkpoint
- `config.toml` and `manifest.json`: run configuration and provenance
- `exports/`: canonical final evaluation, benchmark comparison, hardware
  summary, checkpoint binding, and trace/operator exports
- `analysis/`: post-completion task-family structural recheck over the same
  256 tasks.  It is not a second latency measurement.

## Scope

This is a research artifact for structured graph tasks.  It does not establish
natural-language relation parsing, external-SOTA superiority, or a general
Local LLM knowledge-update system by itself.
