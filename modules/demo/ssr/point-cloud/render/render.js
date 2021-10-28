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

const {RapidsJSDOM}   = require('@rapidsai/jsdom');
const copyFramebuffer = require('./copy')();

class Renderer {
  constructor() {
    const onAnimationFrameRequested = immediateAnimationFrame(this);
    const jsdom                     = new RapidsJSDOM({module, onAnimationFrameRequested});

    const {deck, render} = jsdom.window.evalFn(makeDeck);

    this.deck    = deck;
    this.jsdom   = jsdom;
    this._render = render;
  }
  async render(props = {}, graph = {}, state = {}, events = [], frame = 0) {
    const window = this.jsdom.window;
    return {
      frame: copyFramebuffer(this.deck.animationLoop, frame),
      state: {
        deck: this.deck.serialize(),
        graph:
          this.deck.layerManager.getLayers()?.find((layer) => layer.id === 'PointCloudLayer').graph,
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
        },
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

  const {OrbitView, COORDINATE_SYSTEM} = require('@deck.gl/core');
  const {PointCloudLayer}              = require('@deck.gl/layers');
  const {DeckSSR}                      = require('@rapidsai/deck.gl');

  // Data source: kaarta.com
  const LAZ_SAMPLE =
    'https://raw.githubusercontent.com/uber-common/deck.gl-data/master/examples/point-cloud-laz/indoor.0.1.laz';

  const makeLayers = () => {
    return [
      new PointCloudLayer({
        data: LAZ_SAMPLE,
        coordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
        getNormal: [0, 1, 0],
        getColor: [255, 255, 255],
        opacity: 0.5,
        pointSize: 0.5,
      }),
    ];
  };

  const deck = new DeckSSR({
    createFramebuffer: true,
    initialViewState: {
      target: [0, 0, 0],
      rotationX: 0,
      rotationOrbit: 0,
      orbitAxis: 'Y',
      fov: 50,
      minZoom: 0,
      maxZoom: 10,
      zoom: 1
    },
    layers: makeLayers(),
    views: [
      new OrbitView(),
    ],
    parameters: {clearColor: [0.93, 0.86, 0.81, 1]},
    onAfterAnimationFrameRender({_loop}) { _loop.pause(); },
  });

  return {
    deck,
    render(graph, cb_props = {}) {
      const done = deck.animationLoop.waitForRender();
      deck.setProps({layers: makeLayers(deck, graph).concat([]), ...cb_props});
      deck.animationLoop.start();
      return done;
    },
  };
}
