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

const wrtc     = require('wrtc');
const {nanoid} = require('nanoid');

module.exports.createStream = createStream;

function createStream(peer, id) {
  peer.on('data', onUpdate);
  peer.on('close', onClose);
  peer.on('error', onClose);

  const stream = new wrtc.MediaStream({id: `${id}:video`, isScreencast: true});
  const source = new wrtc.nonstandard.RTCVideoSource({isScreencast: true});
  // console.log(`adding video stream track for id: ${id}`);
  stream.addTrack(source.createTrack());
  peer.addStream(stream);

  function onUpdate(message) {
    if (peer.connected) {
      const {type, data} = (() => {
        try {
          return JSON.parse('' + message);
        } catch (e) { return {}; }
      })();
      // console.log(`onUpdate(${JSON.stringify({type, id, data})})`);
      getWorker().send({
        id,
        type: type || 'render',
        msgId: addMessageListener(onMessage),
        data: {state: getRenderState(id), ...data},
      });
    }
  }

  function onClose() {
    peer.removeListener('data', onUpdate);
    peer.removeListener('close', onClose);
    peer.removeListener('error', onClose);
    if (peer.connected && stream) { peer.removeStream(stream); }
  }

  function onMessage({id, type, ...rest}) {
    // console.log('onMesage:', {id, type, ...rest});
    switch (type) {
      case 'frame': {
        removeMessageListener(id);
        onFrame(rest);
        break;
      }
    }
  }

  function onFrame({frame, state}) {
    if (peer.connected) {
      setRenderState(id, {
        ...getRenderState(id),
        ...state,
        width: frame.width,
        height: frame.height,
      });
      const expectedByteLength = frame.width * frame.height * 3 / 2;
      if (frame.data.buffer.byteLength !== expectedByteLength) {
        frame.data = new Uint8ClampedArray(frame.data.buffer, 0, expectedByteLength).slice();
      }
      source.onFrame(frame);
    }
  }
}

const {addMessageListener, removeMessageListener, dispatchMessage} = (() => {
  const observers = {};
  return {addMessageListener, removeMessageListener, dispatchMessage};
  function addMessageListener(listener) {
    const messageId      = nanoid();
    observers[messageId] = listener;
    return messageId;
  }
  function removeMessageListener(messageId) { delete observers[messageId]; }
  function dispatchMessage(message = {}) {
    const fn = observers[message?.id];
    if (fn) { fn(message); }
  }
})();

const {setRenderState, getRenderState} = (() => {
  const states = {};
  return {setRenderState, getRenderState};
  function setRenderState(id, state) { states[id] = {state}; }
  function getRenderState(id) { return states[id]?.state ?? {}; }
})();

const workers = ((numWorkers = 1) => {
  const workers    = [];
  const Path       = require('path');
  const renderPath = Path.resolve(process.cwd(), 'render');
  const workerPath = Path.join(renderPath, 'index.js');
  const workerOpts = {
    cwd: renderPath,
    serialization: 'advanced',
    stdio: ['pipe', 'inherit', 'inherit', 'ipc'],
    env: {
      ...process.env,
      DISPLAY: undefined,
      WAYLAND_DISPLAY: undefined,
    },
  };

  for (let i = -1; ++i < numWorkers;) {
    workers[i] =
      require('child_process').fork(workerPath, workerOpts).on('message', dispatchMessage);
  }

  process.on('beforeExit', (code = 'SIGKILL') => {
    workers.forEach((worker) => {
      if (!worker.killed) { worker.send({type: 'exit'}); }
      worker.kill(code);
    });
  });
  return workers;
})();

function getWorker() { return workers[(workers.length * Math.random()) | 0]; }
