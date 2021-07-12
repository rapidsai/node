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
import {callMethodSync, callStaticMethodSync} from 'java';

import {
  ArrayList,
  BlazingSchema,
  CatalogColumnImpl,
  CatalogDatabaseImpl,
  CatalogTableImpl,
  RelationalAlgebraGenerator
} from './algebra';
import {Context, default_config} from './context';
import {json_plan_py} from './json_plan';

export class BlazingContext {
  private context: Context;
  private nodes: any[] = [];
  private db: any;
  private schema: any;
  private generator: any;
  private tables: Record<string, DataFrame>;

  constructor() {
    const node: Record<string, unknown> = {};
    node['worker']                      = '';
    this.nodes.push(node);

    this.db        = CatalogDatabaseImpl('main');
    this.schema    = BlazingSchema(this.db);
    this.generator = RelationalAlgebraGenerator(this.schema);
    this.tables    = {};
    this.context   = new Context({
      ralId: 0,
      workerId: 'self',
      network_iface_name: 'lo',
      ralCommunicationPort: 0,
      workersUcpInfo: [],  // TODO: Fix.
      singleNode: true,
      configOptions: default_config,
      allocationMode: 'cuda_memory_resource',
      initialPoolSize: 0,
      maximumPoolSize: null,
      enableLogging: false,
    });
  }

  createTable<T extends TypeMap>(tableName: string, input: DataFrame<T>): void {
    callMethodSync(this.db, 'removeTable', tableName);
    this.tables[tableName] = input;

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

  sql(query: string,
      algebra: string|null                   = null,
      configOptions: Record<string, unknown> = default_config,
      returnToken                            = false) {
    if (algebra == null) { algebra = this.explain(query); }

    if (algebra.includes('LogicalValues(tuples=[[]])') || algebra == '') {
      return new DataFrame({});
    }

    if (algebra.includes(') OVER (')) {
      console.log(
        'WARNING: Window Functions are currently an experimental feature and not fully supported or tested');
    }

    if (returnToken) {
      // TODO: Handle return_token true case.
    }

    const masterIndex       = 0;
    const tableScanInfo     = this.context.getTableScanInfo(algebra);
    const tableNames        = tableScanInfo[0];
    const tableScans        = tableScanInfo[1];
    const d                 = new Date();
    const current_timestamp = `${d.getFullYear()}-${d.getMonth() + 1}-${d.getDate()} ${
      d.getHours()}:${d.getMinutes()}:${d.getSeconds()}.${d.getMilliseconds()}000`;
    const ctxToken = Math.random() * Number.MAX_SAFE_INTEGER;

    const dataframe: DataFrame     = this.tables[tableNames[0]];
    const {names, tables: [table]} = this.context.sql(masterIndex,
                                                      ['self'],
                                                      [dataframe],
                                                      tableNames,
                                                      tableScans,
                                                      ctxToken,
                                                      json_plan_py(algebra),
                                                      configOptions,
                                                      query,
                                                      current_timestamp);

    return new DataFrame(names.reduce(
      (cols, name, i) => ({...cols, [name]: Series.new(table.getColumnByIndex(i))}), {}));
  }

  private explain(sql: string, detail = false): string {
    const algebra = callMethodSync(this.generator, 'getRelationalAlgebraString', sql);

    if (detail == true) {
      // TODO: Handle the true case.
    }

    return String(algebra);
  }
}
