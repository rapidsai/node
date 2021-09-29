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

/* eslint-disable @typescript-eslint/await-thenable */

import {Device} from '@nvidia/cuda';
import {DataFrame} from '@rapidsai/cudf';

import {ContextProps} from './addon';
import {LocalSQLWorker} from './cluster/local';
import {RemoteSQLWorker} from './cluster/remote';
import {defaultClusterConfigValues} from './config';

export interface Worker {
  readonly id: number;
  kill(): void;
  dropTable(name: string): Promise<void>;
  sql(query: string, token: number): Promise<DataFrame>;
  createTable(name: string, table_id: string): Promise<void>;
  createContext(props: Omit<ContextProps, 'id'>): Promise<void>;
}

export interface ClusterProps {
  ip: string;
  port: number;
  numWorkers: number;
}

let ctxToken = 0;

export class SQLCluster {
  /**
   * Initialize and return a new pool of SQLCluster workers.
   *
   * @param options options for the SQLCluster and SQLContext instance(s)
   *
   * @example
   * ```typescript
   * import {SQLCluster} from '@rapidsai/blazingsql';
   *
   * const cluster = await Cluster.init();
   * ```
   */
  public static async init(options: Partial<ClusterProps>&Partial<ContextProps> = {}) {
    const {numWorkers = Device.numDevices, ip = '0.0.0.0', port = 4000} = options;
    const {
      networkIfaceName = 'lo',
      allocationMode   = 'cuda_memory_resource',
      initialPoolSize  = null,
      maximumPoolSize  = null,
      enableLogging    = false,
    }                   = options;
    const configOptions = {...defaultClusterConfigValues, ...options.configOptions};
    const cluster       = new SQLCluster(Math.min(numWorkers, Device.numDevices));
    await cluster._createContexts({
      ip,
      port,
      networkIfaceName,
      allocationMode,
      initialPoolSize,
      maximumPoolSize,
      enableLogging,
      configOptions,
    });
    return cluster;
  }

  private declare _workers: Worker[];
  private declare _worker: LocalSQLWorker;

  private constructor(numWorkers: number) {
    process.on('exit', this.kill.bind(this));
    process.on('beforeExit', this.kill.bind(this));

    this._worker = new LocalSQLWorker(0);
    this._workers =
      Array
        .from({length: numWorkers},
              (_, i) => i === 0
                          ? this._worker
                          : new RemoteSQLWorker(this, i, {...process.env, CUDA_VISIBLE_DEVICES: i}))
        .reverse();
  }

  public get context() { return this._worker.context; }

  protected async _createContexts(props: {ip: string}&Omit<ContextProps, 'id'|'workersUcpInfo'>) {
    const {ip, port}     = props;
    const workersUcpInfo = [...this._workers].reverse().map(({id}) => ({id, ip, port: port + id}));
    await Promise.all(
      this._workers.map((worker) => worker.createContext({...props, workersUcpInfo})));
  }

  /**
   * Create a SQL table to be used for future queries.
   *
   * @param tableName Name of the table when referenced in a query
   * @param input Data source for the table
   *
   * @example
   * ```typescript
   * import {Series, DataFrame, Int32} from '@rapidsai/cudf';
   * import {SQLCluster} from '@rapidsai/blazingsql';
   *
   * const a  = Series.new({type: new Int32(), data: [1, 2, 3]});
   * const b  = Series.new({type: new Int32(), data: [4, 5, 6]});
   * const df = new DataFrame({'a': a, 'b': b});
   *
   * const bc = await SQLCluster.init();
   * await bc.createTable('test_table', df);
   * ```
   */
  public async createTable(tableName: string, input: DataFrame) {
    ctxToken += this._workers.length;
    const ids = this.context.context.broadcast(ctxToken - this._workers.length, input).reverse();
    await Promise.all(this._workers.map((worker, i) => worker.createTable(tableName, ids[i])));
  }

  /**
   * Drop a SQL table from SQLContext memory.
   *
   * @param tableName Name of the table to drop
   *
   * @example
   * ```typescript
   * import {Series, DataFrame, Int32} from '@rapidsai/cudf';
   * import {SQLCluster} from '@rapidsai/blazingsql';
   *
   * const a  = Series.new({type: new Int32(), data: [1, 2, 3]});
   * const b  = Series.new({type: new Int32(), data: [4, 5, 6]});
   * const df = new DataFrame({'a': a, 'b': b});
   *
   * const bc = await SQLCluster.init();
   * await bc.createTable('test_table', df);
   * await bc.dropTable('test_table');
   * console.log(await bc.listTables());
   * // []
   * ```
   */
  public async dropTable(tableName: string) {
    await Promise.all(this._workers.map((worker) => worker.dropTable(tableName)));
  }

  /**
   * Query a SQL table and return the result as a DataFrame.
   *
   * @param query SQL query string
   *
   * @example
   * ```typescript
   * import {Series, DataFrame, Int32} from '@rapidsai/cudf';
   * import {SQLCluster} from '@rapidsai/blazingsql';
   *
   * const a  = Series.new({type: new Int32(), data: [1, 2, 3]});
   * const b  = Series.new({type: new Int32(), data: [4, 5, 6]});
   * const df = new DataFrame({'a': a, 'b': b});
   *
   * const bc = await SQLCluster.init();
   * await bc.createTable('test_table', df);
   *
   * console.log((await bc.sql('SELECT a FROM test_table')).toString())
   * //  a
   * //  0
   * //  1
   * //  2
   * //  3
   * ```
   */
  public async sql(query: string) {
    const algebra = await this.explain(query);
    if (algebra.includes('LogicalValues(tuples=[[]])')) {
      // SQL returns empty result.
      return new DataFrame();
    }
    const token = ctxToken++;
    return new DataFrame().concat(
      ...(await Promise.all(this._workers.map((worker) => worker.sql(query, token)))).reverse());
  }

  /**
   * Returns an array with the names of all created tables.
   *
   * @example
   * ```typescript
   * import {Series, DataFrame, Int32} from '@rapidsai/cudf';
   * import {SQLCluster} from '@rapidsai/blazingsql';
   *
   * const a  = Series.new({type: new Int32(), data: [1, 2, 3]});
   * const df = new DataFrame({'a': a});
   *
   * const bc = await SQLCluster.init();
   * await bc.createTable('test_table', df);
   * console.log(await bc.listTables());
   * // ['test_table']
   * ```
   */
  public listTables() { return this.context.listTables(); }

  /**
   * Returns a map with column names as keys and the column data type as values.
   *
   * @example
   * ```typescript
   * import {Series, DataFrame, Int32} from '@rapidsai/cudf';
   * import {SQLCluster} from '@rapidsai/blazingsql';
   *
   * const a  = Series.new({type: new Int32(), data: [1, 2, 3]});
   * const df = new DataFrame({'a': a});
   *
   * const bc = await SQLCluster.init();
   * await bc.createTable('test_table', df);
   * console.log(await bc.describeTable('test_table'));
   * // {'a': Int32}
   * ```
   */
  public describeTable(tableName: string) { return this.context.describeTable(tableName); }

  /**
   * Returns a break down of a given query's logical relational algebra plan.
   *
   * @param sql SQL query
   * @param detail if a physical plan should be returned instead
   *
   * @example
   * ```typescript
   * import {Series, DataFrame} from '@rapidsai/cudf';
   * import {SQLCluster} from '@rapidsai/blazingsql';
   *
   * const a  = Series.new([1, 2, 3]);
   * const df = new DataFrame({'a': a});
   *
   * const bc = await SQLCluster.init();
   * await bc.createTable('test_table', df);
   *
   * console.log(await bc.explain('SELECT a FROM test_table'));
   * // BindableTableScan(table=[[main, test_table]], aliases=[[a]])
   * ```
   */
  public explain(sql: string, detail = false) { return this.context.explain(sql, detail); }

  /**
   * Sends a `SIGTERM` signal to all spawned workers. Essentially terminates all spawned workers and
   * removes any references to them.
   *
   * @example
   * ```typescript
   * import {SQLCluster} from '@rapidsai/blazingsql';
   *
   * const bc = await SQLCluster.init();
   * bc.kill();
   * ```
   */
  public kill(): void {
    this._workers.forEach((w) => { w.kill(); });
    this._workers.length = 0;
  }
}
