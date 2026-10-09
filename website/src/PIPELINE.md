# Native PreSTO animation

`pipeline-animation.js` adapts the user's `native-excalidraw-renderer/scene.js`.
`public/animation/pipeline.excalidraw` is the original scene used to produce the supplied GIF.

The browser calls the official Excalidraw 0.18.1 `exportToCanvas` API for each frame, at up to 10 frames per second. It retains the source geometry, seeds, bindings, hatching, fonts, IDs and 20-second timeline. Fonts are served locally. Playback pauses when the figure is offscreen, its method tab is hidden, or the page is backgrounded. Reduced-motion users initially see the completed scene, as a static scene.

Build with `npm run build`. `build-animation.mjs` bundles the API and copies its font assets. The remaining page remains plain HTML/CSS/JS. PowerMH's GIF is unchanged.
