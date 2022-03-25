# Server-Side Rendered Visualization App Frontend [WIP]
A front end application connected to the [SSR](https://github.com/rapidsai/node/tree/main/modules/demo/ssr) compute, rendering, and streaming backend. The visualizations are all streamed to a `<video>` tag.

## Featured Dependencies
- React

## Data Requirements
The front end has no explicit data requirements.

## Start
Running the viz-app requires two terminal instances, then opening to `localhost:3000`, and selection your viz type:
```bash
# Terminal 1 - start the rendering server:
yarn demo modules/demo/ssr/graph #/luma /point-cloud

# Terminal 2 - start the front end visualization app:
yarn demo modules/demo/viz-app
```
