# Experiment discipline

The scientific unit is a structured `QuerySpec`, not free-form text. Train, validation, and test generators have separate seeds and persisted split files. The graph generator may provide a valid demonstration trajectory for behavior cloning, but it does not establish an optimal path.

Baselines are fixed traversal, semantic-only retrieval where applicable, and a hand-written hybrid. Metrics are calculated on the same held-out tasks: answer accuracy, proof validity, examined nodes/edges, instructions, ANN calls, credits, latency, and energy when telemetry is available. Inapplicable baselines are reported as such.

Synthetic graphs are explicitly limited to smoke tests. Trial and full modes reject startup without a processed real-data manifest.
