// Copyright (c) 2021, NVIDIA CORPORATION.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
const {IpcMemory, Uint8Buffer} = require('@rapidsai/cuda');

/**
 * makeDeck() returns a Deck and a render callable object to be consumed by the multi-worker
 * Renderer class' JSDOM object
 *
 * @returns {
 *            DeckSSR,
 *            render(layers = {}, boxSelectRectData = [], props = {})
 *          }
 */
function makeDeck() {
  const {log: deckLog} = require('@deck.gl/core');
  deckLog.level        = 0;
  deckLog.enable(false);

  const {OrthographicView}        = require('@deck.gl/core');
  const {TextLayer, PolygonLayer} = require('@deck.gl/layers');
  const {DeckSSR, GraphLayer}     = require('@rapidsai/deck.gl');
  const {OrthographicController}  = require('@rapidsai/deck.gl');

  const makeLayers = (deck, graph = {}) => {
    const [viewport] = (deck?.viewManager?.getViewports() || []);
    const [minX = Number.NEGATIVE_INFINITY,
           minY = Number.NEGATIVE_INFINITY,
    ]                = viewport?.getBounds() || [];
    return [
      new TextLayer({
        sizeScale: 1,
        opacity: 0.9,
        maxWidth: 2000,
        pickable: false,
        getTextAnchor: 'start',
        getAlignmentBaseline: 'top',
        getSize: ({size})          => size,
        getColor: ({color})        => color,
        getPixelOffset: ({offset}) => offset,
        data: Array.from({length: +process.env.NUM_WORKERS},
                         (_, i) =>  //
                         ({
                           size: 15,
                           offset: [0, i * 15],
                           text: `Worker ${i}`,
                           position: [minX, minY],
                           color: +process.env.WORKER_ID === i  //
                                    ? [245, 171, 53, 255]
                                    : [255, 255, 255, 255],
                         }))
      }),
      new GraphLayer({pickable: true, ...graph})
    ];
  };

  const getPolygonLayer = (rectdata) => {
    return new PolygonLayer({
      filled: true,
      stroked: true,
      getPolygon: d => d.polygon,
      lineWidthUnits: 'pixels',
      getLineWidth: 2,
      getLineColor: [80, 80, 80],
      getLineColor: [0, 0, 0, 150],
      getFillColor: [255, 255, 255, 65],
      data: rectdata
    })
  };

  const onDragStart =
    (info, event) => {
      if (deck.props.controller.dragPan) { return; }
      const {x, y}                       = info;
      const [px, py]                     = info.viewport.unproject([x, y]);
      deck.boxSelectCoordinates.startPos = [x, y];
      deck.boxSelectCoordinates.rectdata =
        [{polygon: [[px, py], [px, py], [px, py], [px, py]], show: true}];
    }

  const onDragEnd =
    (info, event) => {
      if (deck.props.controller.dragPan || !deck.boxSelectCoordinates.startPos ||
          !deck.boxSelectCoordinates.rectdata) {
        return;
      }
      const {x, y} = info;
      const sx     = deck.boxSelectCoordinates.startPos[0];
      const sy     = deck.boxSelectCoordinates.startPos[1];

      deck.boxSelectCoordinates.rectdata =
        [{polygon: deck.boxSelectCoordinates.rectdata[0].polygon || [], show: true}];
      deck.boxSelectCoordinates.startPos    = null;
      deck.selectedInfo.selectedCoordinates = {
        x: Math.min(sx, x),
        y: Math.min(sy, y),
        width: Math.abs(x - sx),
        height: Math.abs(y - sy),
        layerIds: ['GraphLayer']
      };

      deck.selectedInfo.nodes = deck.pickObjects(deck.selectedInfo.selectedCoordinates)
                                  .filter(selected => selected.hasOwnProperty('nodeId'))
                                  .map(n => n.nodeId);

      deck.selectedInfo.edges = deck.pickObjects(deck.selectedInfo.selectedCoordinates)
                                  .filter(selected => selected.hasOwnProperty('edgeId'))
                                  .map(n => n.edgeId);
    }

  const onDrag = (info, event) => {
    if (deck.props.controller.dragPan) { return; }
    if (deck.boxSelectCoordinates.startPos) {
      const {x, y}     = info;
      const [px, py]   = info.viewport.unproject([x, y]);
      const startPoint = deck.boxSelectCoordinates.rectdata[0].polygon[0];
      deck.boxSelectCoordinates.rectdata =
        [{polygon: [startPoint, [startPoint[0], py], [px, py], [px, startPoint[1]]], show: true}];
    };
  };

  const onClick = (info, event) => {
    deck.selectedInfo.selectedCoordinates = {
      x: info.x,
      y: info.y,
      radius: 1,
    };
    deck.selectedInfo.nodes = [deck.pickObject(deck.selectedInfo.selectedCoordinates)]
                                .filter(selected => selected && selected.hasOwnProperty('nodeId'))
                                .map(n => n.nodeId);

    console.log(deck.selectedInfo.nodes, deck.selectedInfo.selectedCoordinates);
  };

  const deck = new DeckSSR({
    createFramebuffer: true,
    initialViewState: {
      zoom: 1,
      target: [0, 0, 0],
      minZoom: Number.NEGATIVE_INFINITY,
      maxZoom: Number.POSITIVE_INFINITY,
    },
    layers: [makeLayers(null, {})],
    views: [
      new OrthographicView({
        clear: {
          color: [...[46, 46, 46].map((x) => x / 255), 1],
        },
        controller: {
          keyboard: false,
          doubleClickZoom: false,
          type: OrthographicController,
          scrollZoom: {speed: 0.01, smooth: false},
          // dragPan: false
        }
      }),
    ],
    onAfterAnimationFrameRender({_loop}) { _loop.pause(); },
  });

  deck.selectedInfo         = {selectedCoordinates: {}, selected: []};
  deck.boxSelectCoordinates = {rectdata: [{polygon: [[]], show: false}], startPos: null};
  deck.setProps({onClick, onDrag, onDragStart, onDragEnd});

  return {
    deck,
    render(layers = {}) {
      const done = deck.animationLoop.waitForRender();
      deck.setProps({
        layers: makeLayers(deck, layers)
                  .concat(deck.boxSelectCoordinates.rectdata[0].show
                            ? getPolygonLayer(deck.boxSelectCoordinates.rectdata)
                            : []),
      });
      deck.animationLoop.start();
      return done;
    },
  };
}

function openGraphIpcHandles({nodes, edges, ...graphLayerProps} = {}) {
  const data = {
    nodes: openNodeIpcHandles(nodes),
    edges: openEdgeIpcHandles(edges),
  };
  return {
    pickable: true,
    edgeOpacity: .5,
    edgeStrokeWidth: 2,
    nodesStroked: true,
    nodeFillOpacity: .5,
    nodeStrokeOpacity: .9,
    nodeRadiusScale: 1 / 75,
    nodeRadiusMinPixels: 5,
    nodeRadiusMaxPixels: 150,
    ...graphLayerProps,
    data,
    numNodes: data.nodes.length,
    numEdges: data.edges.length,
  };
}

function closeGraphIpcHandles(graph) {
  closeIpcHandles(graph.data.nodes);
  closeIpcHandles(graph.data.edges);
}

function openNodeIpcHandles(attrs = {}) {
  const attributes = {
    nodeRadius: openIpcHandle(attrs.nodeRadius),
    nodeXPositions: openIpcHandle(attrs.nodeXPositions),
    nodeYPositions: openIpcHandle(attrs.nodeYPositions),
    nodeFillColors: openIpcHandle(attrs.nodeFillColors),
    nodeElementIndices: openIpcHandle(attrs.nodeElementIndices),
  };
  return {offset: 0, length: attrs.length ?? (attributes.nodeRadius?.byteLength || 0), attributes};
}

function openEdgeIpcHandles(attrs = {}) {
  const attributes = {
    edgeList: openIpcHandle(attrs.edgeList),
    edgeColors: openIpcHandle(attrs.edgeColors),
    edgeBundles: openIpcHandle(attrs.edgeBundles),
  };
  return {
    offset: 0,
    length: attrs.length ?? (attributes.edgeList?.byteLength || 0) / 8,
    attributes
  };
}

function openIpcHandle(obj) {
  if (typeof obj === 'string') { obj = JSON.parse(obj); }
  if (obj) {
    const {byteOffset = 0} = obj;
    const handle           = Uint8Array.from(obj.handle.map(Number));
    return new Uint8Buffer(new IpcMemory(handle)).subarray(byteOffset);
  }
  return null;
}

function closeIpcHandles(obj) {
  for (const key in obj) {
    const {buffer} = obj[key] || {};
    if (buffer && (buffer instanceof IpcMemory)) {  //
      buffer.close();
    }
  }
}

function serializeCustomLayer(layers = []) {
  return layers?.find((layer) => layer.id === 'GraphLayer').serialize();
}

module.exports = {
  makeDeck: makeDeck,
  openLayerIpcHandles: openGraphIpcHandles,
  closeLayerIpcHandles: closeGraphIpcHandles,
  serializeCustomLayer: serializeCustomLayer
};
