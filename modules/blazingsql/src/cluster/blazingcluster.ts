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

const CREATE_BLAZING_CONTEXT = 'createBlazingContext';

interface BlazingCusterProps {
  numWorkers: number;
}

export class BlazingCluster {
  workers: ChildProcess[];

  constructor({numWorkers = 1}: BlazingCusterProps) {
    this.workers =
      Array(numWorkers).fill(fork('src/cluster/worker.js', {serialization: 'advanced'}));

    // TODO: Consider a cleaner way to set this up.
    const ucpMetadata = ['0', ...Object.keys(this.workers)].map((_, idx) => ({
                                                                  workerId: idx.toString(),
                                                                  ip: '0.0.0.0',
                                                                  port: 4000 + idx,
                                                                }));

    this.workers.forEach((worker) => worker.send({operation: CREATE_BLAZING_CONTEXT, ucpMetadata}));
  }

  // addTable
  // await sql

  wip() { console.log(this.workers); }
}
