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

import {DataFrame, DataType, Series, TypeMap} from '@rapidsai/cudf';
import {callMethodSync, callStaticMethodSync} from 'java';

import {
  Context,
  getExecuteGraphResult,
  getTableScanInfo,
  runGenerateGraph,
  runGeneratePhysicalGraph,
  startExecuteGraph
} from './addon';
import {
  ArrayList,
  BlazingSchema,
  CatalogColumnImpl,
  CatalogDatabaseImpl,
  CatalogTableImpl,
  RelationalAlgebraGenerator
} from './algebra';
import {defaultConfigValues} from './config';
import {json_plan_py} from './json_plan';

export class BlazingContext {
  // @ts-ignore
  private context: Context;
  private nodes: Record<string, unknown>[] = [];
  private db: any;
  private schema: any;
  private generator: any;
  private tables: Map<string, DataFrame>;

  constructor() {
    const node: Record<string, unknown> = {};
    node['worker']                      = '';
    this.nodes.push(node);

    this.db        = CatalogDatabaseImpl('main');
    this.schema    = BlazingSchema(this.db);
    this.generator = RelationalAlgebraGenerator(this.schema);
    this.tables    = new Map<string, DataFrame>();
    this.context   = new Context({
      ralId: 0,
      workerId: 'self',
      network_iface_name: 'lo',
      ralCommunicationPort: 0,
      workersUcpInfo: [],
      singleNode: true,
      configOptions: defaultConfigValues,
      allocationMode: 'cuda_memory_resource',
      initialPoolSize: 0,
      maximumPoolSize: null,
      enableLogging: false,
    });
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
   * import {BlazingContext} from '@rapidsai/blazingsql';
   *
   * const a  = Series.new({type: new Int32(), data: [1, 2, 3]});
   * const b  = Series.new({type: new Int32(), data: [4, 5, 6]});
   * const df = new DataFrame({'a': a, 'b': b});
   *
   * const bc = new BlazingContext();
   * bc.createTable('test_table', df);
   * ```
   */
  createTable<T extends TypeMap>(tableName: string, input: DataFrame<T>): void {
    callMethodSync(this.db, 'removeTable', tableName);
    this.tables.set(tableName, input);

    const arr = ArrayList();
    input.names.forEach((name: string, index: number) => {
      const dataType =
        callStaticMethodSync('com.blazingdb.calcite.catalog.domain.CatalogColumnDataType',
                             'fromTypeId',
                             input.get(name).type.typeId);
      const column = CatalogColumnImpl([name, dataType, index]);
      callMethodSync(arr, 'add', column);
    });
    const tableJava = CatalogTableImpl([tableName, this.db, arr]);
    callMethodSync(this.db, 'addTable', tableJava);
    this.schema    = BlazingSchema(this.db);
    this.generator = RelationalAlgebraGenerator(this.schema);
  }

  /**
   * Drop a BlazingSQL table from BlazingContext memory.
   *
   * @param tableName Name of the table to drop
   *
   * @example
   * ```typescript
   * import {Series, DataFrame, Int32} from '@rapidsai/cudf';
   * import {BlazingContext} from '@rapidsai/blazingsql';
   *
   * const a  = Series.new({type: new Int32(), data: [1, 2, 3]});
   * const b  = Series.new({type: new Int32(), data: [4, 5, 6]});
   * const df = new DataFrame({'a': a, 'b': b});
   *
   * const bc = new BlazingContext();
   * bc.createTable('test_table', df);
   * bc.sql('SELECT a FROM test_table');
   * bc.dropTable('test_table', df);
   * ```
   */
  dropTable(tableName: string): void {
    if (!this.tables.has(tableName)) {
      throw new Error(
        `Unable to find table with name ${tableName} to drop from BlazingContext memory`);
    }

    callMethodSync(this.db, 'removeTable', tableName);
    this.schema    = BlazingSchema(this.db);
    this.generator = RelationalAlgebraGenerator(this.schema);
    this.tables.delete(tableName);
  }

  /**
   * Returns an array with the names of all created tables.
   *
   * @example
   * ```typescript
   * import {Series, DataFrame, Int32} from '@rapidsai/cudf';
   * import {BlazingContext} from '@rapidsai/blazingsql';
   *
   * const a  = Series.new({type: new Int32(), data: [1, 2, 3]});
   * const df = new DataFrame({'a': a});
   *
   * const bc = new BlazingContext();
   * bc.createTable('test_table', df);
   * bc.listTables(); // ['test_table']
   * ```
   */
  listTables(): string[] { return [...this.tables.keys()]; }

  /**
   * Returns a map with column names as keys and the column data type as values.
   *
   * @example
   * ```typescript
   * import {Series, DataFrame, Int32} from '@rapidsai/cudf';
   * import {BlazingContext} from '@rapidsai/blazingsql';
   *
   * const a  = Series.new({type: new Int32(), data: [1, 2, 3]});
   * const df = new DataFrame({'a': a});
   *
   * const bc = new BlazingContext();
   * bc.createTable('test_table', df);
   * bc.describeTable('test_table'); // {'a': Int32}
   * ```
   */
  describeTable(tableName: string): Map<string, DataType> {
    const table = this.tables.get(tableName);
    return table?.names.reduce(
             (m: Map<string, DataType>, name: string) => m.set(name, table.get(name).type),
             new Map()) ??
           new Map();
  }

  /**
   * Query a BlazingSQL table and return the result as a DataFrame.
   *
   * @param query SQL query string
   * @param algebra SQL algebra plan string, use this to run on a relational algebra query instead
   *   of a query string
   * @param configOptions Set a specific set of configOptions for this query instead of the
   *   defaults
   *
   * @example
   * ```typescript
   * import {Series, DataFrame, Int32} from '@rapidsai/cudf';
   * import {BlazingContext} from '@rapidsai/blazingsql';
   *
   * const a  = Series.new({type: new Int32(), data: [1, 2, 3]});
   * const b  = Series.new({type: new Int32(), data: [4, 5, 6]});
   * const df = new DataFrame({'a': a, 'b': b});
   *
   * const bc = new BlazingContext();
   * bc.createTable('test_table', df);
   *
   * bc.sql('SELECT a FROM test_table'); // [1, 2, 3]
   * ```
   */
  sql(query: string,
      algebra: string|null                   = null,
      configOptions: Record<string, unknown> = defaultConfigValues) {
    if (algebra == null) { algebra = this.explain(query); }

    if (algebra.includes('LogicalValues(tuples=[[]])') || algebra == '') {
      return new DataFrame({});
    }

    if (algebra.includes(') OVER (')) {
      console.log(
        'WARNING: Window Functions are currently an experimental feature and not fully supported or tested');
    }

    const masterIndex      = 0;
    const tableScanInfo    = getTableScanInfo(algebra);
    const tableNames       = tableScanInfo[0];
    const tableScans       = tableScanInfo[1];
    const d                = new Date();
    const currentTimestamp = `${d.getFullYear()}-${d.getMonth() + 1}-${d.getDate()} ${
      d.getHours()}:${d.getMinutes()}:${d.getSeconds()}.${d.getMilliseconds()}000`;
    const ctxToken = Math.random() * Number.MAX_SAFE_INTEGER;
    const selectedDataFrames: DataFrame[] =
      tableNames.reduce((result: DataFrame[], tableName: string) => {
        const table = this.tables.get(tableName);
        if (table !== undefined) { result.push(table); }
        return result;
      }, []);

    const executionGraphResult = runGenerateGraph(masterIndex,
                                                  ['self'],
                                                  selectedDataFrames,
                                                  tableNames,
                                                  tableScans,
                                                  ctxToken,
                                                  json_plan_py(algebra),
                                                  configOptions,
                                                  query,
                                                  currentTimestamp);
    startExecuteGraph(executionGraphResult, ctxToken);

    const {names, tables: [table]} = getExecuteGraphResult(executionGraphResult, ctxToken);
    return new DataFrame(names.reduce(
      (cols, name, i) => ({...cols, [name]: Series.new(table.getColumnByIndex(i))}), {}));
  }

  /**
   * Returns a break down of a given query's logical relational algebra plan.
   *
   * @param sql SQL query
   * @param detail if a physical plan should be returned instead
   *
   * @example
   * ```typescript
   * import {Series, DataFrame from '@rapidsai/cudf';
   * import {BlazingContext} from '@rapidsai/blazingsql';
   *
   * const a  = Series.new([1, 2, 3]);
   * const df = new DataFrame({'a': a});
   *
   * const bc = new BlazingContext();
   * bc.createTable('test_table', df);
   *
   * bc.explain('SELECT a FROM test_table'); // BindableTableScan(table=[[main, test_table]],
   * aliases=[[a]])
   * ```
   */
  explain(sql: string, detail = false): string {
    let algebra = '';

    try {
      algebra = callMethodSync(this.generator, 'getRelationalAlgebraString', sql);

      if (detail == true) {
        const masterIndex = 0;
        const ctxToken    = Math.random() * Number.MAX_SAFE_INTEGER;
        algebra           = json_plan_py(
          runGeneratePhysicalGraph(masterIndex, ['self'], ctxToken, json_plan_py(algebra)), 'True');
      }
    } catch (ex) { throw new Error(ex.cause.getMessageSync()); }

    return String(algebra);
  }
}
