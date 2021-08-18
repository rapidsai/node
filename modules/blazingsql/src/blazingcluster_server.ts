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

import {DataFrame, Series} from '@rapidsai/cudf';
import * as http from 'http';
import * as url from 'url';

import {BlazingCluster} from './blazingcluster';

export class BlazingClusterServer {
  private blazingCluster: BlazingCluster;

  static async init(port = 8888, numWorkers = 2): Promise<BlazingClusterServer> {
    const blazingCluster = await BlazingCluster.init(numWorkers);
    await blazingCluster.createTable('test_table', createLargeDataFrame());

    return new BlazingClusterServer(port, blazingCluster);
  }

  private constructor(port: number, blazingCluster: BlazingCluster) {
    this.blazingCluster = blazingCluster;

    /* eslint-disable @typescript-eslint/no-misused-promises */
    http
      .createServer(async (request: http.IncomingMessage, response: http.ServerResponse) => {
        const path  = url.parse(request.url ?? '', true);
        const query = BlazingClusterServer.parseQueryRequest(path.query);
        response.writeHead(200, 'OK', {'Context-Type': 'text/plain'});

        if (query.length) {
          const result = await this.blazingCluster.sql(query);
          result.names.forEach(
            (n: string) => { response.write(`${n}: ${[...result.get(n)].join()} \n`); });
        }

        response.end(
          '\n\n SQL table "test_table" created. Query this table by adding the following route... \n "/?query="SELECT a FROM test_table"');
      })
      .listen(port);
  }

  private static parseQueryRequest(request: any) {
    const {query} = request;
    return query !== undefined ? query.slice(1, -1) as string : '';
  }
}

// TODO: Load this up with some .csv data
function createLargeDataFrame() {
  const a = Series.new(Array.from(Array(300).keys()));
  return new DataFrame({'a': a, 'b': a});
}
