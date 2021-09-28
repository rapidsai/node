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

import {DataFrame, DataType, TypeMap} from '@rapidsai/cudf';
import {ChildProcess, fork} from 'child_process';

import {ContextProps, UcpContext} from './addon';
import {BlazingContext} from './blazingcontext';
import {defaultClusterConfigValues} from './config';

export const CREATE_BLAZING_CONTEXT  = 'createBlazingContext';
export const BLAZING_CONTEXT_CREATED = 'blazingContextCreated';

export const CREATE_TABLE  = 'createTable';
export const TABLE_CREATED = 'tableCreated';

export const DROP_TABLE    = 'dropTable';
export const TABLE_DROPPED = 'tableDropped';

export const RUN_QUERY = 'runQuery';
export const QUERY_RAN = 'ranQuery';

export type ClusterProps = {
  numWorkers: number,
  ip: string,
  port: number,
};

function _generateMessageId(ctxToken: number): string { return `message_${ctxToken}`; }

let ctxToken = 0;

export class BlazingCluster {
  private workers: ChildProcess[];
  // @ts-ignore
  // instantiated within init()
  private blazingContext: BlazingContext;

  /**
   * Initializes and returns an instance of BlazingCluster.
   *
   * @param clusterOptions optional options for the BlazingCluster instance
   * @param contextConfigOptions optional options for the BlazingContext instance(s)
   *
   * @example
   * ```typescript
   * import {BlazingCluster} from '@rapidsai/blazingsql';
   *
   * await BlazingContext.init();
   * ```
   */
  static async init(clusterOptions: Partial<ClusterProps> = {},
                    contextOptions: Partial<ContextProps> = {}): Promise<BlazingCluster> {
    const {numWorkers = 1, ip = '0.0.0.0', port = 4000} = clusterOptions;
    const {
      id               = 0,
      networkIfaceName = 'lo',
      allocationMode   = 'cuda_memory_resource',
      initialPoolSize  = null,
      maximumPoolSize  = null,
      enableLogging    = false,
    }                   = contextOptions;
    const configOptions = {...defaultClusterConfigValues, ...contextOptions.configOptions};

    const bc = new BlazingCluster(numWorkers - 1);

    const workersUcpInfo = Array.from({length: numWorkers}, (_, id) => ({id, ip, port: port + id}));

    const createContextPromises: Promise<void>[] = [];
    bc.workers.forEach((worker, idx) => {
      createContextPromises.push(new Promise<void>((resolve) => {
        worker.send({
          operation: CREATE_BLAZING_CONTEXT,
          id: id + idx + 1,  // start id at 1 since id 0 is reserved for main process
          port,
          networkIfaceName,
          allocationMode,
          initialPoolSize,
          maximumPoolSize,
          enableLogging,
          workersUcpInfo,
          configOptions,
        });
        worker.once('message', (msg: any) => {
          const {operation}: {operation: string} = msg;
          if (operation === BLAZING_CONTEXT_CREATED) { resolve(); }
        });
      }));
    });

    const ucpContext  = new UcpContext();
    bc.blazingContext = new BlazingContext({
      id,
      port,
      ucpContext,
      networkIfaceName,
      allocationMode,
      initialPoolSize,
      maximumPoolSize,
      enableLogging,
      configOptions,
      workersUcpInfo,
    });

    await Promise.all(createContextPromises);

    return bc;
  }

  private constructor(numWorkers: number) {
    const workerPath =
      __dirname.includes('/src') ? `${__dirname}/../build/js/worker.js` : `${__dirname}/worker.js`;
    this.workers = Array.from({length: numWorkers}, () => fork(workerPath));
  }

  /**
   * Create a BlazingSQL table to be used for future queries.
   *
   * @param tableName Name of the table when referenced in a query
   * @param input Data source for the table
   *
   * @example
   * ```typescript
   * import {Series, DataFrame, Int32} from '@rapidsai/cudf';
   * import {BlazingCluster} from '@rapidsai/blazingsql';
   *
   * const a  = Series.new({type: new Int32(), data: [1, 2, 3]});
   * const b  = Series.new({type: new Int32(), data: [4, 5, 6]});
   * const df = new DataFrame({'a': a, 'b': b});
   *
   * const bc = await BlazingCluster.init();
   * await bc.createTable('test_table', df);
   * ```
   */
  async createTable<T extends TypeMap>(tableName: string, input: DataFrame<T>): Promise<void> {
    const len   = Math.ceil(input.numRows / (this.workers.length + 1));
    const table = input.toArrow();

    const createTablePromises: Promise<void>[] = [];
    this.workers.forEach((worker, i) => {
      const id = i + 1;  // start id at 1 since id 0 is reserved for main process
      ctxToken++;
      const messageId = _generateMessageId(ctxToken);
      createTablePromises.push(new Promise((resolve) => {
        this.blazingContext.send(
          id,
          ctxToken,
          messageId,
          DataFrame.fromArrow(table.slice((i + 1) * len, (i + 2) * len).serialize()));
        worker.send({operation: CREATE_TABLE, tableName: tableName, messageId: messageId});
        worker.once('message', (msg: any) => {
          const {operation}: {operation: string} = msg;
          if (operation === TABLE_CREATED) { resolve(); }
        });
      }));
    });

    this.blazingContext.createTable(tableName,
                                    DataFrame.fromArrow(table.slice(0, len).serialize()));

    await Promise.all(createTablePromises);
  }

  /**
   * Drop a BlazingSQL table from BlazingContext memory.
   *
   * @param tableName Name of the table to drop
   *
   * @example
   * ```typescript
   * import {Series, DataFrame, Int32} from '@rapidsai/cudf';
   * import {BlazingCluster} from '@rapidsai/blazingsql';
   *
   * const a  = Series.new({type: new Int32(), data: [1, 2, 3]});
   * const b  = Series.new({type: new Int32(), data: [4, 5, 6]});
   * const df = new DataFrame({'a': a, 'b': b});
   *
   * const bc = await BlazingCluster.init();
   * await bc.createTable('test_table', df);
   * await bc.dropTable('test_table');
   * ```
   */
  async dropTable(tableName: string): Promise<void> {
    const deleteTablePromises: Promise<void>[] = [];
    this.workers.forEach((worker) => {
      deleteTablePromises.push(new Promise((resolve) => {
        worker.send({operation: DROP_TABLE, tableName: tableName});
        worker.once('message', (msg: any) => {
          const {operation}: {operation: string} = msg;
          if (operation === TABLE_DROPPED) { resolve(); }
        });
      }));
    });

    this.blazingContext.dropTable(tableName);

    await Promise.all(deleteTablePromises);
  }

  /**
   * Returns an array with the names of all created tables.
   *
   * @example
   * ```typescript
   * import {Series, DataFrame, Int32} from '@rapidsai/cudf';
   * import {BlazingCluster} from '@rapidsai/blazingsql';
   *
   * const a  = Series.new({type: new Int32(), data: [1, 2, 3]});
   * const df = new DataFrame({'a': a});
   *
   * const bc = await BlazingCluster.init();
   * await bc.createTable('test_table', df);
   * bc.listTables(); // ['test_table']
   * ```
   */
  listTables(): string[] { return this.blazingContext.listTables(); }

  /**
   * Returns a map with column names as keys and the column data type as values.
   *
   * @example
   * ```typescript
   * import {Series, DataFrame, Int32} from '@rapidsai/cudf';
   * import {BlazingCluster} from '@rapidsai/blazingsql';
   *
   * const a  = Series.new({type: new Int32(), data: [1, 2, 3]});
   * const df = new DataFrame({'a': a});
   *
   * const bc = await BlazingCluster.init();
   * await bc.createTable('test_table', df);
   * bc.describeTable('test_table'); // {'a': Int32}
   * ```
   */
  describeTable(tableName: string): Map<string, DataType> {
    return this.blazingContext.describeTable(tableName);
  }

  /**
   * Returns a break down of a given query's logical relational algebra plan.
   *
   * @param sql SQL query
   * @param detail if a physical plan should be returned instead
   *
   * @example
   * ```typescript
   * import {Series, DataFrame} from '@rapidsai/cudf';
   * import {BlazingCluster} from '@rapidsai/blazingsql';
   *
   * const a  = Series.new([1, 2, 3]);
   * const df = new DataFrame({'a': a});
   *
   * const bc = await BlazingCluster.init();
   * await bc.createTable('test_table', df);
   *
   * bc.explain('SELECT a FROM test_table'); // BindableTableScan(table=[[main, test_table]],
   * aliases=[[a]])
   * ```
   */
  explain(sql: string, detail = false): string { return this.blazingContext.explain(sql, detail); }

  /**
   * Query a BlazingSQL table and return the result as a DataFrame.
   *
   * @param query SQL query string
   *
   * @example
   * ```typescript
   * import {Series, DataFrame, Int32} from '@rapidsai/cudf';
   * import {BlazingCluster} from '@rapidsai/blazingsql';
   *
   * const a  = Series.new({type: new Int32(), data: [1, 2, 3]});
   * const b  = Series.new({type: new Int32(), data: [4, 5, 6]});
   * const df = new DataFrame({'a': a, 'b': b});
   *
   * const bc = await BlazingCluster.init();
   * await bc.createTable('test_table', df);
   *
   * await bc.sql('SELECT a FROM test_table') // [1, 2, 3]
   * ```
   */
  async sql(query: string): Promise<DataFrame> {
    const algebra = this.explain(query);
    if (algebra.includes('LogicalValues(tuples=[[]])')) {
      // SQL returns empty result.
      return new DataFrame();
    }

    const queryPromises = [];
    queryPromises.push(new Promise((resolve) => {
      ctxToken++;
      const messageId = _generateMessageId(ctxToken);
      setTimeout(() => {
        const df = this.blazingContext.sql(query, ctxToken).result();
        resolve({ctxToken: ctxToken, messageId, df});
      });
    }));

    this.workers.forEach((worker) => {
      queryPromises.push(new Promise((resolve) => {
        ctxToken++;
        const messageId = _generateMessageId(ctxToken);
        worker.send({operation: RUN_QUERY, ctxToken: ctxToken, messageId, query});
        worker.once('message', (msg: any) => {
          const {operation, ctxToken, messageId}: {
            operation: string,
            ctxToken: number,
            messageId: string,
          } = msg;

          if (operation === QUERY_RAN) {
            resolve({ctxToken, messageId, df: this.blazingContext.pull(messageId)});
          }
        });
      }));
    });

    let result_df = new DataFrame();

    await Promise.all(queryPromises).then(function(results) {
      results.forEach((result: any) => {
        const {df}: {df: DataFrame} = result;
        result_df                   = result_df.concat(df);
      });
    });

    return result_df;
  }

  /**
   * Sends a `SIGTERM` signal to all spawned workers. Essentially terminates all spawned workers and
   * removes any references to them.
   *
   * @example
   * ```typescript
   * import {BlazingCluster} from '@rapidsai/blazingsql';
   *
   * const bc = await BlazingCluster.init();
   * bc.kill();
   * ```
   */
  kill(): void {
    this.workers.forEach((w) => { w.kill(); });
    this.workers.length = 0;
  }
}
