# NEUROSEEK Jetson low-latency model: final result

日本語版: [FINAL_RESULT.ja.md](FINAL_RESULT.ja.md) · Detailed release notes: [RELEASE_NOTES.md](RELEASE_NOTES.md)

The latency-specialized checkpoint was evaluated on the same fixed 256-task Wikidata5M-based held-out set as its parent, with the same graph, semantic artifact, and CUDA exact backend.

| Metric | Parent | Final model | Change |
| --- | ---: | ---: | ---: |
| Mean end-to-end latency | 31.92 ms | 31.11 ms | -2.56% |
| p95 end-to-end latency | 57.10 ms | 53.02 ms | -7.14% |
| Mean compute credits per task | 233.98 | 146.02 | -37.59% |
| Answer accuracy / valid-proof rate | 98.05% | 97.27% | -0.78 points (2 tasks) |

The final checkpoint reduced latency and search cost, with a small accuracy trade-off. This is a single-Jetson result on one fixed test set. It is not a confidence interval, an external-SOTA comparison, or a natural-language question-answering evaluation. See the [detailed Japanese presentation](report.html) and the canonical exports in `exports/`.
