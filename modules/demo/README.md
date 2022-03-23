# Node RAPIDS Demos
### A collection of demos, templates, and applications showcasing the features and uses for Node RAPIDS.

<br/>

## Starting Demos
In a properly configured env or docker container, most demos can be selected from a list and started with:
```bash
yarn demo
```

Or directly with:
```bash
yarn demo modules/demo/graph
```

Or in most cases within each demo's folder with:
```bash
yarn start
```

## Data Requirements
Some demos require downloading datasets to run, while others have defaults included. See each demo's README for details.

<br/><br/>

# Demos List


## [Viz App](https://github.com/rapidsai/node/tree/main/modules/demo/viz-app) + [SSR Graph](https://github.com/rapidsai/node/tree/main/modules/demo/ssr/graph) | [YouTube Demo](https://youtu.be/zzOCIJ-K1dE)
### Streaming SSR graph visualization with nvENC webRTC to a browser app, using cuDF and cuGraph bindings *Note: future demo will have simplified startup*
![streaming ssr graph app](../../docs/images/demo-screenshots/streaming-graph-demo-ss2-sm.png)

<br/><br/>

## [Viz App](https://github.com/rapidsai/node/tree/main/modules/demo/viz-app) + [SSR Point Cloud](https://github.com/rapidsai/node/tree/main/modules/demo/ssr/graph) | [YouTube Demo](https://youtu.be/vCAiKIkCP3E)
### Streaming SSR point cloud visualization with nvENC webRTC to a browser *Note: future demo will have simplified startup*
![streaming ssr point cloud](../../docs/images/demo-screenshots/streaming-pointcloud-ss-sm.png)

<br/><br/>

## [Client Server](https://github.com/rapidsai/node/tree/main/modules/demo/client-server) | [YouTube Demo](https://youtu.be/H8E0HLiL9YA)
### Browser crossfilter visualization app with server side compute using cuDF
![streaming ssr point cloud](../../docs/images/demo-screenshots/client-server-ss2-sm.png)

<br/><br/>

## [SQL](https://github.com/rapidsai/node/tree/main/modules/demo/sql/sql-cluster-server) | [YouTube Demo](https://youtu.be/EmwcMM_mYKA)
### Demo of multi-GPU SQL demo w/ a browser UI query builder
![streaming ssr point cloud](../../docs/images/demo-screenshots/sql-ss-sm.png)

<br/><br/>

## [Node.js Notebook](https://github.com/rapidsai/node/tree/main/modules/cudf/notebooks) | [YouTube Demo](https://youtu.be/LbHpK8M3cV4)
### GPU accelerated data science in JupyterLab Notebook with Node.js
![streaming ssr point cloud](../../docs/images/demo-screenshots/jupyterlab-nodejs-ss-sm.png)

<br/><br/>

## [Spatial](https://github.com/rapidsai/node/tree/main/modules/demo/spatial)
### Using cuSpatial bindings of Quadtree to segment geospatial data
![Spatial](../../docs/images/demo-screenshots/spatial-ss-sm.png)

<br/><br/>

## [Graph](https://github.com/rapidsai/node/tree/main/modules/demo/graph)
### Using cuGraph bindings of FA2 to compute graph layout
![Spatial](../../docs/images/demo-screenshots/graph-demo-ss-sm.png)

<br/><br/>

## [UMAP](https://github.com/rapidsai/node/tree/main/modules/demo/umap)
### Using cuGraph bindings of UMAP to compute clusters
![UMAP](../../docs/images/demo-screenshots/umap-ss-sm.png)

<br/><br/>

## [Deck.gl](https://github.com/rapidsai/node/tree/main/modules/demo/deck) | [Examples Page](https://deck.gl/examples)
### Running Deck.gl examples with OpenGL Server Side
![Deck.gl](../../docs/images/demo-screenshots/deck-gl-rides-ss-sm.png)

<br/><br/>

## [Luma.gl](https://github.com/rapidsai/node/tree/main/modules/demo/luma) | [Examples Page](https://luma.gl/examples)
### Running (older) Luma.gl examples with OpenGL Server Side
![Luma.gl](../../docs/images/demo-screenshots/luma-ss-sm.png)

<br/><br/>

## Misc Demos
### [Xterm](https://github.com/rapidsai/node/tree/main/modules/demo/xterm) | Emulating GPU rendered xterminal
### [tfjs weblGL](https://github.com/rapidsai/node/tree/main/modules/demo/tfjs/webgl-tests) | ( **Deprecated** ) bindings to tensorflow-js webGl test
### [tfjs RNN addition](https://github.com/rapidsai/node/tree/main/modules/demo/tfjs/addition-rnn) | ( **Deprecated** ) bindings to tensorflow-js addition demo
### [IPC](https://github.com/rapidsai/node/tree/main/modules/demo/ipc) | ( **Deprecated** ) Demo of Inter Process Communication between Python and Node.js


<br/><br/>

# Troubleshooting
## My GLFW window is blank:
If you have more than one GPU, for windowed (no browser) demos you must specify a display GPU by setting: `NVIDIA_VISIBLE_DEVICES=1` in the `.env` file, or docker command. Sometimes it takes a few seconds for things to load too.

## Starting the demo produced an error:
Make sure you have downloaded the required datasets, the data is not malformed, and it is in the correct location -usually `/data`.
If that does not work and if you are running a locally built environment, try rebuilding the module.
Some demo's may have stopped working - check our **[Issues](https://github.com/rapidsai/node/issues)** for details.
