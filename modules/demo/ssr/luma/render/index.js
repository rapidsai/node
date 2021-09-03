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

const Path     = require('path');
const {nanoid} = require('nanoid');
const shm      = require('shm-typed-array');

let numClients        = 0;
const clients         = Object.create(null);
const pendingEvents   = Object.create(null);
const pendingRenders  = Object.create(null);
const pendingResolves = Object.create(null);

function renderEvent(id, event) {
  const queue = pendingEvents[id] || (pendingEvents[id] = []);
  queue.push(event);
  let removed = false;
  return ()   => {
    if (!removed) {
      removed = true;
      const i = queue.findIndex((x) => x === event);
      if (i !== -1) { queue.splice(i, 1); }
    }
  };
}

function renderLesson(data, output) {
  const {id} = data;
  if (!clients[id]) {
    ++numClients;
    clients[id] = {data, output};
  }
  Object.assign(clients[id].data, data);
  start();
  return () => {
    if (clients[id]) {
      --numClients;
      delete clients[id];
      sharedMemoryMap.delete(id);
      if (numClients === 0) { pause(); }
    }
  }
}

module.exports.renderEvent  = renderEvent;
module.exports.renderLesson = renderLesson;

let intervalId = null;

function start() {
  if (intervalId !== null) { return; }

  let workerId = -1;

  intervalId = setInterval(() => {
    for (const clientId in clients) {
      if (pendingRenders[clientId]) { continue; }

      const msgId  = nanoid();
      const {data} = clients[clientId] || {};
      const events = pendingEvents[clientId] || [];
      const buffer = sharedMemoryMap.get(clientId, data);

      delete pendingEvents[clientId];

      pendingRenders[clientId] = new Promise((r) => { pendingResolves[msgId] = r; });

      pendingRenders[clientId].then(({frame, ...data}) => {
        delete pendingRenders[clientId];
        if (clients[clientId]) {
          Object.assign(clients[clientId].data, data);
          clients[clientId].output.onFrame({...frame, data: buffer});
        }
      });

      workerId = (workerId + 1) % workers.length;

      workers[workerId].send({
        id: msgId,
        type: 'render.request',
        data: {data, events, sharedMemoryKey: buffer.key},
      });
    }
  }, 1000 / 60);
}

function pause() {
  if (intervalId !== null) {
    clearInterval(intervalId);
    intervalId = null;
  }
}

const onWorkerExit = (wId, ...xs) => { console.log(`worker ${wId} exit:`, ...xs); };
const onWorkerError = (wId, ...xs) => { console.log(`worker ${wId} error:`, ...xs); };
const onWorkerClose = (wId, ...xs) => { console.log(`worker ${wId} close:`, ...xs); };
const onWorkerDisconnect = (wId, ...xs) => { console.log(`worker ${wId} disconnect:`, ...xs); };
function onWorkerMessage(wId, {id, type, data}) {
  switch (type) {
    case 'render.result': {
      const f = pendingResolves[id];
      delete pendingResolves[id];
      return f && f(data);
    }
  }
}

const sharedMemoryMap = {
  buffersByClientId: new Map(),
  get(id, {width, height}) {
    const map  = this.buffersByClientId;
    const size = width * height * 3 / 2;
    if (!map.has(id)) {  //
      map.set(id, shm.create(size, 'Uint8ClampedArray'));
    } else {
      const mem = map.get(id);
      if (mem.byteLength !== size) {
        shm.detach(mem.key);
        map.set(id, shm.create(size, 'Uint8ClampedArray'));
      }
    }
    return map.get(id);
  },
  delete (id) { this.buffersByClientId.delete(id); }
};

const workers =
  Array
    .from({length: 4}, () => require('child_process').fork(Path.join(__dirname, 'worker.js'), {
      cwd: __dirname,
      execArgv: ['--trace-uncaught'],
      serialization: 'advanced',
      stdio: ['pipe', 'inherit', 'inherit', 'ipc'],
      env: {
        ...process.env,
        DISPLAY: undefined,
        WAYLAND_DISPLAY: undefined,
      },
    }))
    .map(
      (proc, i) => proc.on('exit', onWorkerExit.bind(proc, i))
                     .on('error', onWorkerError.bind(proc, i))
                     .on('close', onWorkerClose.bind(proc, i))
                     .on('message', onWorkerMessage.bind(proc, i))
                     .on('disconnect', onWorkerDisconnect.bind(proc, i))  //
    );

process.on('beforeExit', (code = 'SIGKILL') => {
  workers.forEach((worker) => {
    if (!worker.killed) {
      worker.send({type: 'exit'});
      worker.kill(code);
    }
  });
});

console.log('num workers:', workers.length);
