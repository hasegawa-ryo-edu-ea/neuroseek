**Findings**

- [P1] Browser-rendered visual comparison is unavailable in this execution environment.
  Location: whole WebUI.
  Evidence: source visual truth is `/home/nvidia/.codex/generated_images/019ffc16-82e8-7022-9ba9-fa51cb2b8f52/exec-32e36ab6-e80c-4c29-bf26-45011fb42dfd.png`; this session exposes no browser or screenshot-capture tool for `http://127.0.0.1:8787`.
  Impact: typography, responsive layout, canvas animation, and interaction polish cannot be visually compared to the source image in this environment.
  Fix: open `./search web ja` on the Jetson desktop or through SSH port forwarding, capture the 1472×1027 desktop state, then repeat visual QA against the source image.

**Open Questions**

- The implementation intentionally removes the reference image's unverified GPU/cache/Wikipedia claims. The WebUI instead presents real local graph edges and a real immutable-checkpoint policy run.

**Implementation Checklist**

- [x] Source visual target resolved.
- [x] Bilingual controls, navigation, query input, relation filtering, and policy action are implemented.
- [x] Animated graph canvas represents returned local graph edges, rather than fabricated telemetry.
- [x] CPU/read-only and loopback-only service boundary is enforced.
- [x] HTTP API and primary interactions function-checked through the actual Compose service.
- [ ] Capture and visually compare the rendered UI in a browser.

**Follow-up Polish**

- Compare line density and node-label readability at the presentation display's native resolution after a browser capture is available.

Source visual truth paths: `/home/nvidia/.codex/generated_images/019ffc16-82e8-7022-9ba9-fa51cb2b8f52/exec-32e36ab6-e80c-4c29-bf26-45011fb42dfd.png` and `/home/nvidia/jetson-inference/Final_Project/tmp/スクリーンショット 2026-08-14 072433.png`.

Implementation screenshot path: unavailable — no browser screenshot tool is exposed in this environment.

Viewport: not captured.

State: initial WebUI plus API-verified query (`日本` / `Q17`) and relation filter (`P36`).

Full-view comparison evidence: blocked before capture.

Focused region comparison evidence: blocked before capture.

Primary interactions function-checked: launcher output, loopback health endpoint, Japanese and English entity resolution, local graph view, P36 filter, and held-out policy execution.

Console errors checked: JavaScript syntax was checked with `node --check`; no browser console is available.

Comparison history: the supplied 2557×1377 screenshot was reviewed as the existing rendered baseline. The first iteration added visible drag/zoom/reset affordances, node selection, staged edge/node materialization, and staggered evidence/candidate/result reveals. The second iteration fixes the pointer-start/scene-coordinate mix-up, clamps pan range, increases the graph viewport, renders up to 18 real local edges in a radial graph layout, and exposes node/edge/relation/snapshot/view statistics. The third iteration adds a delayed, selected-branch second-hop expansion sourced from the local graph, with a separate green graph layer and explicit `SECOND HOP · LOCAL ONLY` legend. The fourth iteration adds a `POLICY PATH` control that executes the immutable policy on CPU and replaces the theater with its actual proof path plus alternate frontier words; it was function-verified with Japanese labels for the proof path, relations, and frontier candidates. A current browser capture is still required for pixel-level comparison.

final result: blocked
