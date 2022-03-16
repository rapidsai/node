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
const {makeDeck, openLayerIpcHandles, closeLayerIpcHandles, serializeCustomLayer} =
  require(process.argv[2]);

class Renderer {
  constructor() {
    const onAnimationFrameRequested = immediateAnimationFrame(this);
    const jsdom                     = new RapidsJSDOM({module, onAnimationFrameRequested});

    const {deck, render} = jsdom.window.evalFn(makeDeck);

    this.deck    = deck;
    this.jsdom   = jsdom;
    this._render = render;
  }

  async render(props = {}, layers = {}, state = {}, events = [], frame = 0) {
    const window = this.jsdom.window;

    // get graph layer data, and ipc handles for the gpu buffers
    layers = openLayerIpcHandles(layers);
    state?.layers && Object.assign(layers, state.layers);

    // update deck props as per current state
    props && this.deck.setProps(props);
    state?.deck && this.deck.restore(state.deck);

    // restore current window state
    state?.window && Object.assign(window, state.window);

    // restore current boxSelect state
    state?.selectedInfo && Object.assign(this.deck.selectedInfo, state.selectedInfo);
    state?.boxSelectCoordinates &&
      Object.assign(this.deck.boxSelectCoordinates, state.boxSelectCoordinates);

    // dipatch currently active events in the jsdom window
    (events || []).forEach((event) => window.dispatchEvent(event));

    // render the deck.gl frame using the current layers
    await this._render(layers,
                       this.deck.boxSelectCoordinates.rectdata,
                       state.pickingMode === 'boxSelect' ? {controller: {dragPan: false}}
                                                         : {controller: {dragPan: true}});

    // close the ipc handles for layer data
    closeLayerIpcHandles(layers);

    // return the frame to the main process along with the current state
    return {
      frame: copyFramebuffer(this.deck.animationLoop, frame),
      state: {
        deck: this.deck.serialize(),
        layers: serializeCustomLayer(this.deck.layerManager.getLayers()),
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
        boxSelectCoordinates: this.deck.boxSelectCoordinates,
        selectedInfo: this.deck.selectedInfo
      }
    };
  }
}

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

module.exports.Renderer = Renderer;
