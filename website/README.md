# PreSTO project website

Static site for [PreSTO](https://github.com/Nan-Jiang-Group/PreSTO).


The separate `PreSTO-website` repository is retired; make all website changes here.



## Source layout

| Path | Contents |
| --- | --- |
| `public/index.html` | Page content and sections. |
| `public/*.css` | Styles (`taste`, `experiments`, `edge-certainty`). |
| `public/figure-one.js`, `case-studies.js`, `edge-certainty.js` | Interactive experiment plots. |
| `public/data/` | Exported paper measurements and prompt scores. |
| `public/app.js`, `public/simulation.mjs` | Synthetic tree explorer. |
| `src/pipeline-animation.js`, `public/animation/pipeline.excalidraw` | Method animation. |
| `scripts/` | Optional data exporters; they point to the author's local research checkout and are not needed to build. |
| `tests/` | Scheduling, data, and chart tests. |

## Build and hosting

`npm run build` writes the static site to `dist/`; any static host works. `.openai/hosting.json` links the source to its Sites project (no credentials). Pushing to GitHub does not publish the site or change its access. Sites access is owner-only.

Generated bundles, dependencies, and `dist/` are not tracked in Git. Excalidraw and font licenses ship with their assets.

## Tree explorer semantics

- **Depth:** MH decisions along a complete path. The root counts as a proposal; terminal states do not.
- **Budget:** number of scored decision nodes (not leaf batches, tokens, or GPU work).
- **Schedules:** ancestor-closed. Optional cut feasibility requires an accept child to have `c′ ≤ c`; reject children are unrestricted.
- **Effective transitions:** consecutive scheduled decisions along one sampled path, up to the first unscheduled node. Rejections count.
- **Expected transitions:** sum of the scheduled nodes' exact reach probabilities. Schedules see synthetic probabilities and cuts, never the uniforms that realize the path.
- **STeP-inspired preset (synthetic):** 41% p=0, 9% p=.005, 17% p∈[.1,.9), 1% p=.995, 32% p=1. Defaults: depth 5, budget 8, tree seed 17, path seed 17.

These are illustrative scheduling conventions, not a production engine or a speedup benchmark.


