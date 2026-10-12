import {build} from 'esbuild';
import {cpSync,mkdirSync,rmSync} from 'node:fs';
rmSync('public/animation-build',{recursive:true,force:true});
await build({entryPoints:['src/pipeline-animation.js'],bundle:true,minify:true,format:'esm',splitting:true,outdir:'public/animation-build',define:{'process.env.NODE_ENV':'"production"'},loader:{'.woff2':'file','.woff':'file','.ttf':'file','.png':'file','.svg':'file'},logLevel:'warning'});
mkdirSync('public/excalidraw-assets',{recursive:true});
cpSync('node_modules/@excalidraw/excalidraw/dist/prod/fonts','public/excalidraw-assets/fonts',{recursive:true});
