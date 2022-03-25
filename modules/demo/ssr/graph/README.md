# Graph Server Side Rendering (SSR) and Streaming Server
The back end to the [Viz-App](https://github.com/rapidsai/node/tree/main/modules/demo/viz-app) demo, using cuGraph FA2, luma.gl for visualization, and default graph data. Streamed using webRTC utilizing [nvENC](https://docs.nvidia.com/video-technologies/video-codec-sdk/nvenc-video-encoder-api-prog-guide/).

## Featured Dependencies
- @rapidsai/cudf
- @rapidsai/cugraph
- @rapidsai/jsdom
- @rapidsai/deckgl

## Data Requirements
The graph demo will default to internal data.

## Start
To start the server:
```bash
yarn start #NOTE: For the demo to work, run this in one terminal instance AND run the Viz-App in another terminal (use graph option)
```
