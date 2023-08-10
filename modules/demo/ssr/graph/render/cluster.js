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

const workerPath = require('path').join(__dirname, 'worker.js');

class RenderCluster {
  constructor({fps = 60, numWorkers = 4} = {}) {
    this._fps     = fps;
    this._timerId = null;
    this._reqs    = Object.create(null);
    this._jobs    = Object.create(null);
    this._workers = this._createWorkers(numWorkers);

    process.on('exit', killWorkers);
    process.on('beforeExit', killWorkers);

    function killWorkers(code = 'SIGKILL') {
      this._workers.forEach((worker) => {
        if (!worker.killed) {
          worker.send({type: 'exit', code});
          worker.kill(code);
        }
      });
    }
  }

  pause() {
    if (this._timerId !== null) {
      clearInterval(this._timerId);
      this._timerId = null;
    }
  }

  start() {
    if (this._timerId === null) {
      this._timerId = setInterval(() => this.flush(), 1000 / this._fps);
    }
  }

  render(id, data, callback) {
    const request = this._reqs[id] || (this._reqs[id] = Object.create(null));
    (request.callbacks || (request.callbacks = [])).push(callback);
    request.data || (request.data = Object.create(null));
    Object.assign(request.data, data);
    this.start();
  }

  flush() {
    const requests = this._reqs;
    this._reqs     = Object.create(null);
    const workers  = this._workers.slice().sort((a, b) => a.jobs - b.jobs);
    let workerId   = 0;
    for (const id in requests) {
      if (!(id in this._jobs)) {
        this._jobs[id] = dispatchJob(id, requests, workers[workerId]);
        workerId       = (workerId + 1) % workers.length;
      }
    }
  }

  _onWorkerExit(workerId, ...xs) { console.log(`worker ${workerId} exit`, ...xs); }

  _onWorkerError(workerId, ...xs) { console.log(`worker ${workerId} error`, ...xs); }

  _onWorkerClose(workerId, ...xs) { console.log(`worker ${workerId} close`, ...xs); }

  _onWorkerDisconnect(workerId, ...xs) { console.log(`worker ${workerId} disconnect`, ...xs); }

  _onWorkerMessage(_workerId, {id, type, data}) {
    switch (type) {
      case 'render.result':
        const p = this._jobs[id];
        delete this._jobs[id];
        return p && p.resolve(data);
    }
  }

  _createWorkers(numWorkers = 4) {
    return Array.from({length: numWorkers}).map((_, i) => {
      const worker = require('child_process').fork(workerPath, {
        cwd: __dirname,
        // serialization: 'advanced',
        execArgv: ['--trace-uncaught'],
        stdio: ['pipe', 'inherit', 'inherit', 'ipc'],
        env: {
          ...process.env,
          DISPLAY: undefined,
          WORKER_ID: i,
          NUM_WORKERS: numWorkers,
        },
      });

      worker.jobs = 0;

      return worker.on('exit', this._onWorkerExit.bind(this, i))
        .on('error', this._onWorkerError.bind(this, i))
        .on('close', this._onWorkerClose.bind(this, i))
        .on('message', this._onWorkerMessage.bind(this, i))
        .on('disconnect', this._onWorkerDisconnect.bind(this, i));
    });
  }
}

function dispatchJob(id, requests, worker) {
  const {promise, resolve, reject} = promiseSubject();

  promise
    .catch((err) => {
      if (id in requests) {
        --worker.jobs;
        const {callbacks} = requests[id];
        delete requests[id];
        callbacks.forEach((cb) => cb(err, null));
      }
    })
    .then((data) => {
      if (id in requests) {
        --worker.jobs;
        const {callbacks} = requests[id];
        delete requests[id];
        callbacks.forEach((cb) => cb(null, data));
      }
    });

  const data = {...requests[id].data};
  ++worker.jobs;
  worker.send({id, type: 'render.request', data});

  return {promise, resolve, reject};
}

function promiseSubject() {
  let resolve, reject;
  let promise = new Promise((r1, r2) => {
    resolve = r1;
    reject  = r2;
  });
  return {promise, resolve, reject};
}

module.exports.RenderCluster = RenderCluster;
