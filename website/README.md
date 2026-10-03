# PreSTO project website

Website source within the canonical [Nan-Jiang-Group/PreSTO](https://github.com/Nan-Jiang-Group/PreSTO) repository, with interactive benchmark plots, Appendix E.2 likelihood and confidence box plots, case studies, a synthetic tree explorer, and a native Excalidraw animation.

Website: https://step-tree-explorer.jiangnanhugo.chatgpt.site/

The separate `PreSTO-website` repository is retired; update this `website/` directory for future changes.

## Run locally

Install Node.js and Python 3, then run:

```sh
npm ci
npm run build
npm start
```

Open http://localhost:4173/. The build bundles the Excalidraw API and its fonts before serving the page. No model calls or research logs are needed to view the website: the exported experiment data is included.

```sh
npm test
```

## Source layout

- `public/index.html`: page content and section layout.
- `public/taste.css`, `public/experiments.css`, `public/edge-certainty.css`: styling.
- `public/figure-one.js`, `public/case-studies.js`, `public/edge-certainty.js`: interactive experiment plots.
- `public/data/`: exported paper measurements and prompt scores.
- `public/app.js`, `public/simulation.mjs`: synthetic tree explorer.
- `src/pipeline-animation.js`, `public/animation/pipeline.excalidraw`: native animated method diagram.
- `scripts/`: optional data exporters; these currently refer to the author's local research checkout and are unnecessary for building or viewing the included data.
- `tests/`: scheduling, data, and chart validation.

## Build and hosting

`npm run build` produces the static website in `dist/`. Serve that directory with a static host. The `.openai/hosting.json` file connects this source to its existing Sites project; it contains no credentials. Source synchronization to GitHub does not automatically publish the website or change its access settings.

Generated bundles, dependencies, and `dist/` are excluded from Git. Excalidraw and font license notices are included with their respective assets.

## Semantics

- Depth is the number of MH decisions along a complete path. The root counts as a proposal; terminal states do not.
- Budget counts scored decision nodes, not leaf batches, tokens, or GPU work.
- Schedules are ancestor-closed. Optional local cut feasibility requires an accept child to have c′ ≤ c. Reject children have no extra local restriction.
- Effective transitions are the consecutive scheduled decisions on one common sampled path, stopping at the first unscheduled node. Rejections count.
- Expected transitions are the sum of scheduled nodes' exact reach probabilities. Scheduling sees synthetic probabilities and cuts, but never the random uniforms used to realize the path.
- The STeP-inspired preset is synthetic: 41% p=0, 9% p=.005, 17% p∈[.1,.9), 1% p=.995, 32% p=1. Mixture weights are not exact finite-tree counts. Default depth 5, budget 8, tree seed 17, path seed 17.
- These illustrative scheduling conventions are documented in the interface; this is not a reproduction of a production STeP engine or a speedup benchmark.

Configuration is stored only in this browser's localStorage. There are no analytics, external scripts, remote fonts, or model calls. Sites enforces owner-only access at the hosting layer.

## Validation

The test suite verifies deterministic schedule order, budget bounds, ancestor closure, cut feasibility, p=0 and p=1 cases, full coverage, deterministic replay, absence of path-draw leakage into schedules, and exact expected transitions against exhaustive path enumeration.
