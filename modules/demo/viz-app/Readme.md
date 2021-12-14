# Server-Side Rendered Visualization App Frontend [WIP]
A collection of react-based visualizations, such as geospatil, graph, and scatterplot, all rendered server-side and streamed to a simple <video> tag via a custom version of webRTC using nvENC. For more details visit the [SSR demos section](https://github.com/rapidsai/node/tree/main/modules/demo/ssr)

## Featured Dependencies
deckgl


## Data Requirements
NOTE: while the frontend has no explicit data requirements, it assumes the [SSR Graph demo](https://github.com/rapidsai/node/tree/main/modules/demo/ssr/graph) is server running.

## Start
Running visualization app requires two running docker instances, then opening `localhost:3000/`:
`
# Terminal 1 - start the rendering server:
yarn demo modules/demo/ssr/graph

# Terminal 2 - start the front end visualization app:
yarn demo modules/demo/viz-app
`