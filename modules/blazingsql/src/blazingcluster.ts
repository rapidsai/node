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

import {ChildProcess, fork} from 'child_process';
import {UcpContext} from './addon';
import {BlazingContext} from './blazingcontext';

export const CREATE_BLAZING_CONTEXT = 'createBlazingContext';
export const CONFIG_OPTIONS         = {
  PROTOCOL: 'UCX',
};

interface BlazingCusterProps {
  numWorkers: number;
}

export class BlazingCluster {
  workers: ChildProcess[];
  bc: BlazingContext;

  constructor({numWorkers = 1}: BlazingCusterProps) {
    this.workers = Array(numWorkers).fill(fork(`${__dirname}/worker`, {serialization: 'advanced'}));

    // TODO: Consider a cleaner way to set this up.
    const ucpMetadata = ['0', ...Object.keys(this.workers)].map(
      (_, idx) => { return ({workerId: idx.toString(), ip: '0.0.0.0', port: 4000 + idx}); });

    this.workers.forEach((worker, idx) => {
      const ralId = idx + 1;  // start ralId at 1 since ralId 0 is reserved for main process
      worker.send({operation: CREATE_BLAZING_CONTEXT, ralId, ucpMetadata});
    });

    const ucpContext = new UcpContext();
    this.bc          = new BlazingContext({
      ralId: 0,
      ralCommunicationPort: 4000,
      configOptions: {...CONFIG_OPTIONS},
      workersUcpInfo: ucpMetadata.map((xs) => ({...xs, ucpContext})),
    });
  }

  // addTable
  // await sql
}
