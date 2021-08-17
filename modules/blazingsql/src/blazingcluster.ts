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

import {DataFrame, Series, TypeMap} from '@rapidsai/cudf';
import {ChildProcess, fork} from 'child_process';

import {UcpContext} from './addon';
import {BlazingContext} from './blazingcontext';

export const CREATE_BLAZING_CONTEXT  = 'createBlazingContext';
export const BLAZING_CONTEXT_CREATED = 'blazingContextCreated';

export const CREATE_TABLE  = 'createTable';
export const TABLE_CREATED = 'tableCreated';

export const RUN_QUERY = 'runQuery';
export const QUERY_RAN = 'ranQuery';

export const CONFIG_OPTIONS = {
  PROTOCOL: 'UCX',
};

interface BlazingCusterProps {
  numWorkers: number;
}

export class BlazingCluster {
  private workers: ChildProcess[];
  // @ts-ignore
  // instantiated within init()
  private blazingContext: BlazingContext;

  static async init(args: Partial<BlazingCusterProps>): Promise<BlazingCluster> {
    const bc         = new BlazingCluster(args);
    const ucpContext = new UcpContext();

    const ucpMetadata = ['0', ...Object.keys(bc.workers)].map(
      (_, idx) => { return ({workerId: idx.toString(), ip: '0.0.0.0', port: 4000 + idx}); });

    const createContextPromises: Promise<void>[] = [];
    bc.workers.forEach((worker, idx) => {
      createContextPromises.push(new Promise<void>((resolve) => {
        const ralId = idx + 1;  // start ralId at 1 since ralId 0 is reserved for main process
        worker.send({operation: CREATE_BLAZING_CONTEXT, ralId, ucpMetadata});
        worker.on('message', (msg: any) => {
          const {operation}: {operation: string} = msg;
          if (operation === BLAZING_CONTEXT_CREATED) { resolve(); }
        });
      }));
    });

    bc.blazingContext = new BlazingContext({
      ralId: 0,
      ralCommunicationPort: 4000,
      configOptions: {...CONFIG_OPTIONS},
      workersUcpInfo: ucpMetadata.map((xs) => ({...xs, ucpContext})),
    });

    await Promise.all(createContextPromises);

    return bc;
  }

  private constructor(args: Partial<BlazingCusterProps>) {
    const {numWorkers = 1} = args;

    this.workers = Array(numWorkers);
    for (let i = 0; i < numWorkers; ++i) {
      this.workers[i] = fork(`${__dirname}/worker`, {serialization: 'advanced'});
    }
  }

  async createTable<T extends TypeMap>(tableName: string, input: DataFrame<T>): Promise<void> {
    const len   = Math.ceil(input.numRows / (this.workers.length + 1));
    const table = input.toArrow();

    const createTablePromises: Promise<void>[] = [];
    this.workers.forEach((worker, i) => {
      const ralId = i + 1;  // start ralId at 1 since ralId 0 is reserved for main process
      createTablePromises.push(new Promise((resolve) => {
        this.blazingContext.sendToCache(
          ralId,
          ralId,
          `message_${ralId}`,
          DataFrame.fromArrow(table.slice((i + 1) * len, (i + 2) * len).serialize()));
        worker.send({operation: CREATE_TABLE, tableName: tableName, ralId: ralId});
        worker.on('message', (msg: any) => {
          const {operation}: {operation: string} = msg;
          if (operation === TABLE_CREATED) { resolve(); }
        });
      }));
    });

    this.blazingContext.createTable(tableName,
                                    DataFrame.fromArrow(table.slice(0, len).serialize()));

    await Promise.all(createTablePromises);
  }

  async sql(query: string): Promise<DataFrame> {
    let ctxToken        = 0;
    const queryPromises = [];

    queryPromises.push(new Promise((resolve) => {
      const token     = ctxToken++;
      const messageId = `message_${token}`;
      setTimeout(() => {
        const df = this.blazingContext.sql(query, token).result();
        resolve({ctxToken: token, messageId, df});
      });
    }));

    this.workers.forEach((worker) => {
      queryPromises.push(new Promise((resolve) => {
        const token     = ctxToken++;
        const messageId = `message_${token}`;
        worker.send({operation: RUN_QUERY, ctxToken: token, messageId, query});
        worker.on('message', (msg: any) => {
          const {operation, ctxToken, messageId}: {
            operation: string,
            ctxToken: number,
            messageId: string,
          } = msg;

          if (operation === QUERY_RAN) {
            resolve({ctxToken, messageId, df: this.blazingContext.pullFromCache(messageId)});
          }
        });
      }));
    });

    let result_df = new DataFrame({a: Series.new([])});

    await Promise.all(queryPromises).then(function(results) {
      results.forEach((result: any) => {
        const {df}: {df: DataFrame} = result;
        result_df                   = result_df.concat(df);
      });
    });

    return result_df;
  }

  stop(): void {
    this.workers.forEach((worker) => { worker.kill('SIGKILL'); });
  }
}
