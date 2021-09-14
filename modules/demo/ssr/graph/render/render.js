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

const {IpcMemory, Uint8Buffer} = require('@nvidia/cuda');
const {RapidsJSDOM}            = require('@rapidsai/jsdom');
const copyFramebuffer          = require('./copy')();

class Renderer {
  constructor() {
    const onAnimationFrameRequested = immediateAnimationFrame(this);
    const jsdom                     = new RapidsJSDOM({module, onAnimationFrameRequested});
    const {deck, render}            = jsdom.window.evalFn(makeDeck);

    this.deck    = deck;
    this.jsdom   = jsdom;
    this._render = render;
  }
  async render(props = {}, graph = {}, state = {}, events = [], frame = 0) {
    graph = openGraphIpcHandles(graph);
    props && this.deck.setProps(props);
    state?.deck && this.deck.restore(state.deck);
    state?.graph && Object.assign(graph, state.graph);
    state?.window && Object.assign(this.jsdom.window, state.window);

    await this._render(graph);

    if (events.length > 0) {
      events.forEach((event) => {  //
        // if (event.type === 'wheel') { debugger; }
        this.jsdom.window.dispatchEvent(event);
      });
      await this._render(graph);
    }

    closeIpcHandles(graph.data.nodes);
    closeIpcHandles(graph.data.edges);

    return {
      frame: frame > 0 ? {...copyFramebuffer(this.deck.animationLoop, frame)}
                       : {width: state.width, height: state.height, frame},
      state: {
        deck: this.deck.serialize(),
        graph: this.deck.layerManager.getLayers()[0]?.serialize(),
        window: {
          x: this.jsdom.window.x,
          y: this.jsdom.window.y,
          title: this.jsdom.window.title,
          width: this.jsdom.window.width,
          height: this.jsdom.window.height,
          cursor: this.jsdom.window.cursor,
          mouseX: this.jsdom.window.mouseX,
          mouseY: this.jsdom.window.mouseY,
          buttons: this.jsdom.window.buttons,
          scrollX: this.jsdom.window.scrollX,
          scrollY: this.jsdom.window.scrollY,
          modifiers: this.jsdom.window.modifiers,
        },
      },
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

  const {OrthographicView, OrthographicController} = require('@deck.gl/core');

  class ImmediateOrthographicController extends OrthographicController {
    get linearTransitionProps() { return null; }
  }

  const {DeckSSR, GraphLayer} = require('@rapidsai/deck.gl');
  const deck                  = new DeckSSR({
    createFramebuffer: true,
    controller: {
      keyboard: false,
      type: ImmediateOrthographicController,
      scrollZoom: {speed: 0.01, smooth: false},
    },
    initialViewState: {
      zoom: 1,
      target: [0, 0, 0],
      minZoom: Number.NEGATIVE_INFINITY,
      maxZoom: Number.POSITIVE_INFINITY,
    },
    layers: [new GraphLayer({pickable: true})],
    views: [
      new OrthographicView({
        clear: {
          color: [...[46, 46, 46].map((x) => x / 255), 1],
        }
      }),
    ],
    onAfterAnimationFrameRender({_loop}) { _loop.pause(); },
  });

  return {
    deck,
    render(graph = {}) {
      const rendered = deck.animationLoop.waitForRender();
      deck.setProps({layers: [new GraphLayer(graph)]});
      deck.animationLoop.start();
      return rendered;
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
