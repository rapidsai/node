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

require('@babel/register')({
  cache: false,
  babelrc: false,
  cwd: __dirname,
  presets: [
    ['@babel/preset-env', {'targets': {'node': 'current'}}],
    ['@babel/preset-react', {'useBuiltIns': true}]
  ]
});

process.chdir(__dirname);

const {RapidsJSDOM} = require('@rapidsai/jsdom');

let timeout;
let animationFrameRequest = null;
const jsdom               = new RapidsJSDOM({
  // onAnimationFrameRequested(r) {
  //   // animationFrameRequest = r;
  //   if (!timeout) {
  //     timeout = setTimeout(() => {
  //       timeout = null;
  //       r.flush();
  //     }, 0);
  //   }
  //   jsdom.window.poll();
  // }
});

let triggers = Object.create(null);

process.on('message', ({id, type, data, msgId} = {}) => {
  switch (type) {
    case 'exit': return process.exit();
    default:
      // console.log(`process.on('message', (`, {id, type, data, msgId}, `)`);
      const updates = triggers[id] || (triggers[id] = Object.create(null));
      const queue   = updates[type] || (updates[type] = []);
      queue.push([msgId, data]);
      break;
  }
});

setInterval(async () => {
  const runAll = async (fn, queue = []) => {
    for (const args of queue) {
      const p = fn(...args);
      await Promise.resolve();
      flushAnimationFrame();
      await p;
    }
  };
  const triggers_ = triggers;
  triggers        = Object.create(null);
  for (const id in triggers_) {
    const {render: renderQueue, event: eventQueue} = triggers_[id];
    if (renderQueue) { await runAll(handleRender, renderQueue); }
    if (eventQueue) { await runAll(handleDispatch, eventQueue); }
  }
}, 1000 / 60);

function flushAnimationFrame() {
  if (animationFrameRequest) {
    const {flush}         = animationFrameRequest;
    animationFrameRequest = null;
    flush(() => {});
  }
}

function handleRender(msgId, {json = {}, state = {}, ...props}) {
  let resolve, promise = new Promise((r) => resolve = r);

  props = {initialViewState: {}, ...state, ...props};

  // console.log(`handleRender`, {msgId, ...props});

  if (typeof props.width === 'number') { jsdom.window.width = props.width; }
  if (typeof props.height === 'number') { jsdom.window.height = props.height; }

  try {
    jsdom.window.evalFn(renderFrame, {
      __json: json,
      __props: props,
      __onAfterRender: onAfterRender,
    });
  } catch (e) { console.error(e); }

  return promise;

  function onAfterRender({viewState, interactionState, frame}) {
    const {width, height, data} = frame;
    const state                 = {...props, json, viewState, interactionState};
    // console.log('onAfterRender',
    //             {state, frame: {width, height, data: {byteLength: data.byteLength}}});
    process.send({id: msgId, type: 'frame', state, frame});
    resolve();
  }

  function renderFrame() {
    const {createElement} = require('react');
    const {render}        = require('react-dom');
    const {converter}     = require('../configuration');
    const DeckGL          = require('@deck.gl/react').default;
    const props           = {
      ...__props,
      ...converter.convert(__json),
      ...require(`./props`)(__onAfterRender)
    };
    const [root = document.body.appendChild(document.createElement('div'))] =
      document.body.childNodes;
    render(createElement(DeckGL, props), root);
  }
}

function handleDispatch(msgId, {state = {}, ...event}) {
  // console.log(`handleDispatch`, {msgId, state, event});
  jsdom.window.dispatchEvent(event);
  jsdom.window.poll();
  return handleRender(msgId, state);
}
