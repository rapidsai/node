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

import {DataFrame, Table} from '@rapidsai/cudf';

export class ExecutionGraph {
  constructor(private _graph?: import('./rapidsai_sql').ExecutionGraph) {}

  start(): void { this._graph?.start(); }

  async result() {
    const {names, tables} =
      this._graph ? (await this._graph.result()) : {names: [], tables: [new Table({})]};
    const results: DataFrame[] = [];
    tables.forEach((table: Table) => {
      results.push(new DataFrame(
        names.reduce((cols, name, i) => ({...cols, [name]: table.getColumnByIndex(i)}), {})));
    });

    return results;
  }

  async sendTo(id: number) { return await this.result().then((df) => this._graph?.sendTo(id, df)); }
}
