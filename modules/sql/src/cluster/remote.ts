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

/* eslint-disable @typescript-eslint/no-unused-vars */
/* eslint-disable @typescript-eslint/await-thenable */

import {ChildProcess, fork} from 'child_process';
import {nanoid} from 'nanoid';
import * as Path from 'path';

import {ContextProps} from '../addon';

import {SQLCluster, Worker} from '../cluster';

const remoteWorkerPath =
  Path.resolve(__dirname.includes('/src') ? `${__dirname}/../../build/js/cluster/worker.js`
                                          : `${__dirname}/worker.js`);

export class RemoteSQLWorker implements Worker {
  declare public readonly id: number;

  declare private _proc: ChildProcess;

  declare private _jobs: Map<string, {
    promise: Promise<any>,
    resolve: (val: any) => any,
    reject: (err?: any) => any,
  }>;

  declare private _cluster: SQLCluster;

  constructor(cluster: SQLCluster, id: number, env: any = {...process.env}) {
    this.id       = id;
    this._jobs    = new Map();
    this._cluster = cluster;
    this._proc    = fork(remoteWorkerPath, {env})
                   .on('message', this._recv.bind(this))
                   .on('exit', this._onExit.bind(this))
                   .on('error', this._onError.bind(this))
                   .on('close', this._onClose.bind(this))
                   .on('disconnect', this._onDisconnect.bind(this));
  }

  public kill() {
    if (this._connected) {
      this._proc.send({type: 'exit', code: 0});
      this._proc.kill();
    }
  }

  public createContext(props: Omit<ContextProps, 'id'>) {
    const {id} = this, port = props.port + id;
    return this._send({type: 'init', ...props, id, port}).then(() => undefined);
  }

  public createTable(name: string, table_id: string) {
    return this._send({type: 'createTable', name, table_id}).then(() => undefined);
  }

  public createCSVTable(name: string, paths: string[]) {
    return this._send({type: 'createCSVTable', name, paths}).then(() => undefined);
  }

  public dropTable(name: string) {
    return this._send({type: 'dropTable', name}).then(() => undefined);
  }

  public sql(query: string, token: number) {
    return this._send({type: 'sql', query, token})
      .then(({messageIds}) =>
              Promise.all(messageIds.map((id: string) => this._cluster.context.pull(id))));
  }

  private _send({type, ...rest}: any = {}) {
    if (this._connected) {
      const uuid = nanoid();
      this._jobs.set(uuid, promiseSubject());
      this._proc.send({type, uuid, ...rest});
      // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
      return this._jobs.get(uuid)!.promise;
    }
    return Promise.resolve({});
  }

  private _recv({error, ...rest}: any = {}) {
    const {uuid} = rest;
    if (uuid && this._jobs.has(uuid)) {
      // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
      const {resolve, reject} = this._jobs.get(uuid)!;
      this._jobs.delete(uuid);
      (error != null) ? reject(error) : resolve(rest);
    }
  }

  private get _connected() { return this._proc && !this._proc.killed; }

  private _onExit(..._: any[]) {
    // console.log(`worker ${this.id} exit`, ..._);
  }
  private _onError(..._: any[]) {
    // console.log(`worker ${this.id} error`, ..._);
  }
  private _onClose(..._: any[]) {
    // console.log(`worker ${this.id} close`, ..._);
  }
  private _onDisconnect(..._: any[]) {
    // console.log(`worker ${this.id} disconnect`, ..._);
  }
}

function promiseSubject() {
  let resolve = (_x: any) => {};
  let reject = (_er: any) => {};
  const promise           = new Promise((r1, r2) => {
    resolve = r1;
    reject  = r2;
  });
  return {promise, resolve, reject};
}
