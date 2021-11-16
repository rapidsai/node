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

import {ContextProps, UcpContext} from '../addon';
import {Worker} from '../cluster';
import {SQLContext} from '../context';

export class LocalSQLWorker implements Worker {
  declare public readonly id: number;
  declare public context: SQLContext;

  constructor(id: number) { this.id = id; }

  public kill() {}

  public createContext(props: Omit<ContextProps, 'id'>) {
    return new Promise<void>((resolve) => {
      const {id} = this, port = props.port + id;
      this.context = new SQLContext({...props, id, port, ucpContext: new UcpContext});
      resolve();
    });
  }

  public async createDataFrameTable(name: string, table_id: string) {
    const table = await this.context.pull(table_id);
    this.context.createDataFrameTable(name, table);
  }

  public async createCSVTable(name: string, paths: string[]) {
    await this.context.createCSVTable(name, paths);
  }

  public async createParquetTable(name: string, paths: string[]) {
    await this.context.createParquetTable(name, paths);
  }

  public async createORCTable(name: string, paths: string[]) {
    await this.context.createORCTable(name, paths);
  }

  public async dropTable(name: string) { await this.context.dropTable(name); }

  public async sql(query: string, token: number) {
    return await (await this.context.sql(query, token)).result();
  }
}
