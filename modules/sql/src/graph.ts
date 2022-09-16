// Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

let nonce = Math.random() * 1e3 | 0;

export class ExecutionGraph implements Promise<DataFrame[]> {
  constructor(private _graph?: import('./rapidsai_sql').ExecutionGraph) {}

  get[Symbol.toStringTag]() { return 'ExecutionGraph'; }

  then<TResult1 = DataFrame[], TResult2 = never>(
    onfulfilled?: ((value: DataFrame[]) => TResult1 | PromiseLike<TResult1>)|undefined|null,
    onrejected?: ((reason: any) => TResult2 | PromiseLike<TResult2>)|undefined|
    null): Promise<TResult1|TResult2> {
    return this.result().then(onfulfilled, onrejected);
  }

  catch<TResult = never>(onrejected?: ((reason: any) => TResult | PromiseLike<TResult>)|undefined|
                         null): Promise<DataFrame[]|TResult> {
    return this.result().catch(onrejected);
  }

  finally(onfinally?: (() => void)|undefined|null): Promise<DataFrame[]> {
    return this.result().finally(onfinally);
  }

  private _result: Promise<DataFrame[]>|undefined;

  start() { this._graph?.start(); }

  result() {
    if (!this._result) {
      this._result = (async () => {
        const {names, tables} =
          this._graph ? (await this._graph.result()) : {names: [], tables: [new Table({})]};
        const results: DataFrame[] = [];
        tables.forEach((table: Table) => {
          results.push(new DataFrame(
            names.reduce((cols, name, i) => ({...cols, [name]: table.getColumnByIndex(i)}), {})));
        });

        return results;
      })();
    }
    return this._result;
  }

  sendTo(id: number) {
    return this.then((dfs) => {
      const {_graph}                                  = this;
      const inFlightTables: Record<string, DataFrame> = {};
      if (_graph) {
        _graph.sendTo(id, dfs, `${nonce++}`).forEach((messageId, i) => {  //
          inFlightTables[messageId] = dfs[i];
        });
      }
      return inFlightTables;
    });
  }
}
