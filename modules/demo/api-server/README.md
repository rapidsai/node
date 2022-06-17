# Fastify HTTP Server Demo

This project is a Fastify-based node http server that allows
commands and data to be sent to and from the NVIDIA GPUs installed
on the host machine.

Essentially, the node-rapids system is provided as a backend to any
HTTP client. At this time only limited functionality is available to
load JSON files in the `graphology` graph dataset format, plus API
requests to request Dataframes and their Columns via `apache-arrow`.

Two endpoints, `graphology/nodes` and `graphology/edges` specifically
return pre-formatted arrays that can be used directly with the
[sigma.js](https://github.com/jacomyal/sigma.js) renderer. An
[extra-large-graphs](https://github.com/jacomyal/sigma.js/pull/1252) example PR is in the works
that utilizes this GPU-accelerated data for rendering larger datasets
than available via only CPU.

## Main Dependencies
- @rapidsai/cudf
- fastify
- fastify-arrow
- apache-arrow

An example project that demonstrates this API has a PR being reviewed at [sigma.js](https://github.com/jacomyal/sigma.js),
but this project does not depend on sigma.js.j

## Installation

To install dependencies, run the following from the root directory for `node-rapids`

```bash
yarn
```

To run the demo
```bash
# Select the api-server demo from the list of demos
yarn demo
# OR specifically with
cd modules/demo/api-server
yarn start
```

## Dataset

Run the graph generator at https://github.com/thomcom/sigma.js/blob/add-gpu-graph-to-example/examples/extra-large-graphs/generate-graph.js
to create a very large graph using the object

```js
const state = {
  order: 1000000,
  size: 2000000,
  clusters: 3,
  edgesRenderer: 'edges-fast'
};
```

You don't need to edit the file in order to create a graph of the above size. Simply call the .js via node:

```bash
node graph-generator.js
```

Which will create a file `./large-graph.json`. Copy `./large-graph.json` into `api-server/` and then set your
API request to the location of the file relative to `routes/graphology/index.js`:

```
curl http://localhost:3000/graphology/read_large_demo?filename=../../large-graph.json
```

Which will use parallel JSON parsing to load the graph onto the GPU.

## Routes

```txt
/
/graphology
/graphology/read_json
/graphology/read_large_demo
/graphology/list_tables
/graphology/get_table/:table
/graphology/get_column/:table/:column
/graphology/nodes/bounds
/graphology/nodes
/graphology/edges
/graphology/release
