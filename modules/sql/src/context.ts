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

import {DataFrame, DataType} from '@rapidsai/cudf';
import {UcpContext} from '.';

import {
  Context,
  ContextProps,
  getTableScanInfo,
  runGeneratePhysicalGraph,
} from './addon';
import {
  ArrayList,
  BlazingSchema,
  CatalogColumnDataType,
  CatalogColumnImpl,
  CatalogDatabaseImpl,
  CatalogTableImpl,
  RelationalAlgebraGenerator
} from './algebra';
import {defaultContextConfigValues} from './config';
import {ExecutionGraph} from './graph';
import {json_plan_py} from './json_plan';
import {DataFrameTable, FileTable, SQLTable} from './SQLTable';

export class SQLContext {
  public readonly context: Context;
  declare private _db: any;
  declare private _schema: any;
  declare private _generator: any;
  declare private _ucpContext?: UcpContext;
  declare private _tables: Map<string, SQLTable>;
  declare private _configOptions: typeof defaultContextConfigValues;

  constructor(options: Partial<ContextProps> = {}) {
    this._db        = CatalogDatabaseImpl('main');
    this._schema    = BlazingSchema(this._db);
    this._generator = RelationalAlgebraGenerator(this._schema);
    this._tables    = new Map<string, SQLTable>();

    const {
      id               = 0,
      port             = 0,
      networkIfaceName = 'lo',
      workersUcpInfo   = [],
      allocationMode   = 'cuda_memory_resource',
      initialPoolSize  = null,
      maximumPoolSize  = null,
      enableLogging    = false,
      ucpContext,
    } = options;

    this._ucpContext    = ucpContext;
    this._configOptions = {...defaultContextConfigValues, ...options.configOptions};

    this.context = new Context({
      id,
      port,
      networkIfaceName,
      ucpContext,
      workersUcpInfo,
      configOptions: this._configOptions,
      allocationMode,
      initialPoolSize,
      maximumPoolSize,
      enableLogging
    });
  }

  public get id() { return this.context.id; }

  /**
   * Create a SQL table from cudf.DataFrames.
   *
   * @param tableName Name of the table when referenced in a query
   * @param input cudf.DataFrame
   *
   * @example
   * ```typescript
   * import {Series, DataFrame, Int32} from '@rapidsai/cudf';
   * import {SQLContext} from '@rapidsai/sql';
   *
   * const a  = Series.new({type: new Int32(), data: [1, 2, 3]});
   * const b  = Series.new({type: new Int32(), data: [4, 5, 6]});
   * const df = new DataFrame({'a': a, 'b': b});
   *
   * const sqlContext = new SQLContext();
   * sqlContext.createDataFrameTable('test_table', df);
   * ```
   */
  createDataFrameTable(tableName: string, input: DataFrame): void {
    this._createTable(new DataFrameTable(tableName, input));
  }

  /**
   * Create a SQL table from CSV file(s).
   *
   * @param tableName Name of the table when referenced in a query
   * @param filePaths array of paths to CSV file(s)
   *
   * @example
   * ```typescript
   * import {SQLContext} from '@rapidsai/sql';
   *
   * const sqlContext = new SQLContext();
   * sqlContext.createCSVTable('test_table', ['test.csv']);
   * ```
   */
  createCSVTable(tableName: string, filePaths: string[]): void {
    this._createTable(new FileTable(tableName, filePaths, 'csv'));
  }

  /**
   * Create a SQL table from Apache Parquet file(s).
   *
   * @param tableName Name of the table when referenced in a query
   * @param filePaths array of paths to parquet file(s)
   *
   * @example
   * ```typescript
   * import {SQLContext} from '@rapidsai/sql';
   *
   * const sqlContext = new SQLContext();
   * sqlContext.createParquetTable('test_table', ['test.parquet']);
   * ```
   */
  createParquetTable(tableName: string, filePaths: string[]): void {
    this._createTable(new FileTable(tableName, filePaths, 'parquet'));
  }

  /**
   * Create a SQL table from Apache ORC file(s).
   *
   * @param tableName Name of the table when referenced in a query
   * @param filePaths array of paths to ORC file(s)
   *
   * @example
   * ```typescript
   * import {SQLContext} from '@rapidsai/sql';
   *
   * const sqlContext = new SQLContext();
   * sqlContext.createORCTable('test_table', ['test.orc']);
   * ```
   */
  createORCTable(tableName: string, filePaths: string[]): void {
    this._createTable(new FileTable(tableName, filePaths, 'orc'));
  }

  private _createTable(input: SQLTable): void {
    if (this._tables.has(input.tableName)) {  //
      this._db.removeTableSync(input.tableName);
    }
    this._tables.set(input.tableName, input);

    const arr = ArrayList();
    input.names.forEach((name: string, index: number) => {
      const dataType = CatalogColumnDataType.fromTypeIdSync(input.type(name).typeId);
      const column   = CatalogColumnImpl([name, dataType, index]);
      arr.addSync(column);
    });
    const tableJava = CatalogTableImpl([input.tableName, this._db, arr]);
    this._db.addTableSync(tableJava);
    this._schema    = BlazingSchema(this._db);
    this._generator = RelationalAlgebraGenerator(this._schema);
  }

  /**
   * Drop a SQL table from SQLContext memory.
   *
   * @param tableName Name of the table to drop
   *
   * @example
   * ```typescript
   * import {Series, DataFrame, Int32} from '@rapidsai/cudf';
   * import {SQLContext} from '@rapidsai/sql';
   *
   * const a  = Series.new({type: new Int32(), data: [1, 2, 3]});
   * const b  = Series.new({type: new Int32(), data: [4, 5, 6]});
   * const df = new DataFrame({'a': a, 'b': b});
   *
   * const sqlContext = new SQLContext();
   * sqlContext.createTable('test_table', df);
   * sqlContext.sql('SELECT a FROM test_table');
   * sqlContext.dropTable('test_table', df);
   * ```
   */
  public dropTable(tableName: string): void {
    if (!this._tables.has(tableName)) {
      throw new Error(`Unable to find table with name ${tableName} to drop from SQLContext memory`);
    }

    this._db.removeTableSync(tableName);
    this._schema    = BlazingSchema(this._db);
    this._generator = RelationalAlgebraGenerator(this._schema);
    this._tables.delete(tableName);
  }

  /**
   * Returns an array with the names of all created tables.
   *
   * @example
   * ```typescript
   * import {Series, DataFrame, Int32} from '@rapidsai/cudf';
   * import {SQLContext} from '@rapidsai/sql';
   *
   * const a  = Series.new({type: new Int32(), data: [1, 2, 3]});
   * const df = new DataFrame({'a': a});
   *
   * const sqlContext = new SQLContext();
   * sqlContext.createTable('test_table', df);
   * sqlContext.listTables(); // ['test_table']
   * ```
   */
  public listTables(): string[] { return [...this._tables.keys()]; }

  /**
   * Returns a map with column names as keys and the column data type as values.
   *
   * @example
   * ```typescript
   * import {Series, DataFrame, Int32} from '@rapidsai/cudf';
   * import {SQLContext} from '@rapidsai/sql';
   *
   * const a  = Series.new({type: new Int32(), data: [1, 2, 3]});
   * const df = new DataFrame({'a': a});
   *
   * const sqlContext = new SQLContext();
   * sqlContext.createTable('test_table', df);
   * sqlContext.describeTable('test_table'); // {'a': Int32}
   * ```
   */
  public describeTable(tableName: string): Map<string, DataType> {
    const table = this._tables.get(tableName);
    if (table === undefined) { return new Map(); }
    return table.names.reduce(
      (m: Map<string, DataType>, name: string) => m.set(name, table.type(name)), new Map());
  }

  /**
   * Query a SQL table and return the result as a DataFrame.
   *
   * @param query SQL query string
   * @param ctxToken an optional content token used for communicating multiple nodes
   *
   * @example
   * ```typescript
   * import {Series, DataFrame, Int32} from '@rapidsai/cudf';
   * import {SQLContext} from '@rapidsai/sql';
   *
   * const a  = Series.new({type: new Int32(), data: [1, 2, 3]});
   * const b  = Series.new({type: new Int32(), data: [4, 5, 6]});
   * const df = new DataFrame({'a': a, 'b': b});
   *
   * const sqlContext = new SQLContext();
   * sqlContext.createTable('test_table', df);
   *
   * await sqlContext.sql('SELECT a FROM test_table'); // [1, 2, 3]
   * ```
   */
  public sql(query: string, ctxToken: number = Math.random() * Number.MAX_SAFE_INTEGER | 0) {
    const algebra = this.explain(query);
    if (algebra == '') { throw new Error('ERROR: Failed to parse given query'); }

    if (algebra.includes('LogicalValues(tuples=[[]])')) {
      // SQL returns an empty execution graph.
      return new ExecutionGraph();
    }

    if (algebra.includes(') OVER (')) {
      console.log(
        'WARNING: Window Functions are currently an experimental feature and not fully supported or tested');
    }

    const tableScanInfo    = getTableScanInfo(algebra);
    const tableNames       = tableScanInfo[0];
    const tableScans       = tableScanInfo[1];
    const d                = new Date();
    const currentTimestamp = `${d.getFullYear()}-${d.getMonth() + 1}-${d.getDate()} ${
      d.getHours()}:${d.getMinutes()}:${d.getSeconds()}.${d.getMilliseconds()}000`;

    const selectedDataFrames: DataFrame[]            = [];
    const selectedSchemas: Record<string, unknown>[] = [];
    tableNames.forEach((tableName: string) => {
      const table = this._tables.get(tableName);
      if (table !== undefined) {
        if (table instanceof DataFrameTable) {
          selectedDataFrames.push(table.getSource());
        } else {
          selectedSchemas.push(table.getSource());
        }
      }
    });

    return new ExecutionGraph(this.context.runGenerateGraph(selectedDataFrames,
                                                            selectedSchemas,
                                                            tableNames,
                                                            tableScans,
                                                            ctxToken,
                                                            json_plan_py(algebra),
                                                            this._configOptions,
                                                            query,
                                                            currentTimestamp));
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
   * import {SQLContext} from '@rapidsai/sql';
   *
   * const a  = Series.new([1, 2, 3]);
   * const df = new DataFrame({'a': a});
   *
   * const sqlContext = new SQLContext();
   * sqlContext.createTable('test_table', df);
   *
   * sqlContext.explain('SELECT a FROM test_table'); // BindableTableScan(table=[[main,
   * test_table]], aliases=[[a]])
   * ```
   */
  public explain(sql: string, detail = false): string {
    let algebra = '';

    try {
      algebra = this._generator.getRelationalAlgebraStringSync(sql);

      if (detail == true) {
        const ctxToken = Math.random() * Number.MAX_SAFE_INTEGER;
        algebra =
          json_plan_py(runGeneratePhysicalGraph(['self'], ctxToken, json_plan_py(algebra)), 'True');
      }
    } catch (ex: any) { throw new Error(ex.cause.getMessageSync()); }

    return String(algebra);
  }

  /**
   * Sends a DataFrame to the cache machine.
   *
   * @param id The id of the destination SQLContext
   * @param ctxToken The token associated with the messageId
   * @param messageId The id used to pull the table on the destination SQLContext
   *
   * @example
   * ```typescript
   * import {Series, DataFrame from '@rapidsai/cudf';
   * import {SQLContext} from '@rapidsai/sql';
   *
   * const a  = Series.new([1, 2, 3]);
   * const df = new DataFrame({'a': a});
   *
   * const sqlContext = new SQLContext();
   * sqlContext.send(0, 0, "message_1", df);
   * ```
   */
  public send(id: number, ctxToken: number, messageId: string, df: DataFrame) {
    this.context.send(id, ctxToken, messageId, df);
  }

  /**
   * Returns a DataFrame pulled from the cache machine.
   *
   * @param messageId The message id given when initially sending the DataFrame to the cache
   *
   * @example
   * ```typescript
   * import {Series, DataFrame from '@rapidsai/cudf';
   * import {SQLContext} from '@rapidsai/sql';
   *
   * const a  = Series.new([1, 2, 3]);
   * const df = new DataFrame({'a': a});
   *
   * const sqlContext = new SQLContext();
   * sqlContext.send(0, 0, "message_1", df);
   * await sqlContext.pull("message_1"); // [1, 2, 3]
   * ```
   */
  async pull(messageId: string) {
    const {names, tables: [table]} = await this.context.pull(messageId);
    return new DataFrame(
      names.reduce((cols, name, i) => ({...cols, [name]: table.getColumnByIndex(i)}), {}));
  }
}
