// Copyright (c) 2021-2023, NVIDIA CORPORATION.
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
const {RapidsJSDOM}            = require('@rapidsai/jsdom');
const copyFramebuffer          = require('./copy')();

class Renderer {
  constructor() {
    const onAnimationFrameRequested = immediateAnimationFrame(this);
    const jsdom                     = new RapidsJSDOM({
      module,
      onAnimationFrameRequested,
      babel: {
        presets: [
          // transpile all ESM to CJS
          ['@babel/preset-env', {targets: {node: 'current'}}],
          ...RapidsJSDOM.defaultOptions.babel.presets,
        ]
      },
    });

    const {deck, render} = jsdom.window.evalFn(makeDeck);

    this.deck    = deck;
    this.jsdom   = jsdom;
    this._render = render;
  }
  async render(props = {}, graph = {}, state = {}, events = [], frame = 0) {
    const window = this.jsdom.window;

    graph = openGraphIpcHandles(graph);
    props && this.deck.setProps(props);

    state?.deck && this.deck.restore(state.deck);
    state?.graph && Object.assign(graph, state.graph);
    state?.window && Object.assign(window, state.window);

    (events || []).forEach((event) => window.dispatchEvent(event));

    await this._render(graph);

    closeIpcHandles(graph.data.nodes);
    closeIpcHandles(graph.data.edges);

    const deck = this.deck.serialize();

    return {
      frame: copyFramebuffer(this.deck.animationLoop, frame),
      state: {
        ...state,
        deck: {
          ...deck,
          props: {
            ...deck.props,
            boxSelection: this.deck?.props?.boxSelection,
            selectedNodes: this.deck?.props?.selectedNodes,
            selectedEdges: this.deck?.props?.selectedEdges,
          }
        },
        graph: this.deck.layerManager.getLayers()
                 ?.find((layer) => layer.id === 'GraphLayer')
                 .serialize(),
        window: {
          x: window.x,
          y: window.y,
          title: window.title,
          width: window.width,
          height: window.height,
          cursor: window.cursor,
          mouseX: window.mouseX,
          mouseY: window.mouseY,
          buttons: window.buttons,
          scrollX: window.scrollX,
          scrollY: window.scrollY,
          modifiers: window.modifiers,
          mouseInWindow: window.mouseInWindow,
        }
      }
    };
  }
}

module.exports.Renderer = Renderer;

function immediateAnimationFrame(renderer) {
  let request  = null;
  let flushing = false;
  const flush = () => {
    flushing = true;
    while (request && request.active) {
      const f = request.flush;
      request = null;
      f();
    }
    flushing = false;
  };
  return (r) => {
    if (flushing) { return request = r; }
    if (renderer?.deck?.animationLoop?._initialized) {  //
      return flush(request = r);
    }
    if (!request && (request = r)) { setImmediate(flush); }
  };
}

function makeDeck() {
  const {log: deckLog} = require('@deck.gl/core');
  deckLog.level        = 0;
  deckLog.enable(false);

  const {OrthographicView}        = require('@deck.gl/core');
  const {TextLayer, PolygonLayer} = require('@deck.gl/layers');
  const {DeckSSR, GraphLayer}     = require('@rapidsai/deck.gl');

  const makeLayers = (deck, graph = {}) => {
    const boxSelection = deck?.props?.boxSelection;
    const [viewport]   = (deck?.viewManager?.getViewports() || []);
    const [minX = Number.NEGATIVE_INFINITY,
           minY = Number.NEGATIVE_INFINITY,
    ]                  = viewport?.getBounds() || [];
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
      new GraphLayer({pickable: true, ...graph}),
      new PolygonLayer({
        filled: true,
        stroked: true,
        getPolygon: d => d.polygon,
        lineWidthUnits: 'pixels',
        getLineWidth: 2,
        getLineColor: [80, 80, 80],
        getLineColor: [0, 0, 0, 150],
        getFillColor: [255, 255, 255, 65],
        data: boxSelection ? [boxSelection] : []
      })
    ];
  };

  function onDragStart(info, event) {
    if (event?.srcEvent?.shiftKey) {
      event.preventDefault();
      const {x, y}   = info;
      const [px, py] = info.viewport.unproject([x, y]);
      deck.setProps({
        boxSelection: {
          origin: [x, y],
          polygon: [[px, py], [px, py], [px, py], [px, py]],
        }
      });
      return true;
    }
  }

  function onDrag(info, event) {
    if (deck.props.boxSelection) {
      event.preventDefault();
      const {x, y}     = info;
      const [px, py]   = info.viewport.unproject([x, y]);
      const [[sx, sy]] = deck.props.boxSelection.polygon;
      deck.setProps({
        boxSelection: {
          ...deck.props.boxSelection,
          polygon: [[sx, sy], [sx, py], [px, py], [px, sy]],
        }
      });
      return true;
    }
  };

  function onDragEnd(info, event) {
    if (deck.props.boxSelection) {
      event.preventDefault();

      const {x, y}   = info;
      const [sx, sy] = deck.props.boxSelection.origin;

      const selections = deck.pickObjects({
        x: Math.min(sx, x),
        y: Math.min(sy, y),
        width: Math.abs(x - sx),
        height: Math.abs(y - sy),
        layerIds: ['GraphLayer']
      });

      deck.setProps({
        boxSelection: null,
        selectedNodes: selections.filter(x => x.hasOwnProperty('nodeId')).map(n => n.nodeId),
        selectedEdges: selections.filter(x => x.hasOwnProperty('edgeId')).map(n => n.edgeId),
      });

      if (deck.props.selectedNodes.length > 0) {
        console.log({'selected nodes': deck.props.selectedNodes});
      }
      if (deck.props.selectedEdges.length > 0) {
        console.log({'selected edges': deck.props.selectedEdges});
      }
      return true;
    }
  }

  function onClick(info, event) {
    const selections = deck.pickObjects({
      x: info.x,
      y: info.y,
      radius: 1,
    });

    deck.setProps({
      boxSelection: null,
      selectedNodes: selections.filter(x => x.hasOwnProperty('nodeId')).map(n => n.nodeId),
      selectedEdges: selections.filter(x => x.hasOwnProperty('edgeId')).map(n => n.edgeId),
    });

    if (deck.props.selectedNodes.length > 0) {
      console.log({'selected nodes': deck.props.selectedNodes});
    }
    if (deck.props.selectedEdges.length > 0) {
      console.log({'selected edges': deck.props.selectedEdges});
    }
  }

  const deck = new DeckSSR({
    _sync: true,
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
          scrollZoom: {speed: 0.01, smooth: false},
        }
      }),
    ],
    onAfterAnimationFrameRender({_loop}) { _loop.pause(); },
    onClick,
    onDrag,
    onDragStart,
    onDragEnd,
    selectedNodes: [],
    selectedEdges: [],
  });

  return {
    deck,
    render(graph) {
      const done = deck.animationLoop.waitForRender();
      deck.setProps({layers: makeLayers(deck, graph)});
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

function openNodeIpcHandles(attrs = {}) {
  const positions  = openIpcHandle(attrs.positions);
  const attributes = {
    nodeRadius: openIpcHandle(attrs.nodeRadius),
    nodeXPositions: positions.subarray(0, positions.length / 2),
    nodeYPositions: positions.subarray(positions.length / 2),
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
