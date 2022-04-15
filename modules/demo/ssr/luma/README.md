# Luma.gl Server Side Rendering (SSR) and Streaming Server
The back end to the [Viz-App](https://github.com/rapidsai/node/tree/main/modules/demo/viz-app) demo, using luma.gl for rendering. Streamed using webRTC utilizing [nvENC](https://docs.nvidia.com/video-technologies/video-codec-sdk/nvenc-video-encoder-api-prog-guide/).

## Featured Dependencies
- @rapidsai/jsdom
- @rapidsai/deckgl

## Data Requirements
The demo will default to internal data, a pretty little teapot.

## Start
To start the server:
```bash
yarn start #NOTE: For the demo to work, run this in one terminal instance AND run the Viz-App in another terminal (use point cloud option)
```
