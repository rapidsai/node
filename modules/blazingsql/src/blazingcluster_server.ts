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

import {BlazingCluster} from './blazingcluster';

export class BlazingClusterServer {
  // @ts-ignore
  private server: http.Server;
  // @ts-ignore
  private blazingCluster: BlazingCluster;

  static async init(port = 8888, numWorkers = 2): Promise<BlazingClusterServer> {
    const blazingCluster = await BlazingCluster.init(numWorkers);
    await blazingCluster.createTable('test', createLargeDataFrame());

    return new BlazingClusterServer(port, blazingCluster);
  }

  private constructor(port: number, blazingCluster: BlazingCluster) {
    this.blazingCluster = blazingCluster;

    /* eslint-disable @typescript-eslint/no-misused-promises */
    this.server =
      http
        .createServer((request: http.IncomingMessage, response: http.ServerResponse) => {
          const {url}       = request;
          const body: any[] = [];
          request.on('error', (err) => { console.error(err); })
            .on('data', (chunk) => { body.push(chunk); })
            .on('end', async () => {
              response.writeHead(200);
              const query = BlazingClusterServer.parseQuery(url);
              if (query.length !== 0) {
                const result = await blazingCluster.sql(BlazingClusterServer.parseQuery(url));
                response.end([...result.get('a')].join());
              } else {
                response.end('Please enter a query');
              }
            });
        })
        .listen(port);
  }

  private static parseQuery(url: string|undefined) {
    return url?.replace(/\W+/g, '').replaceAll('_', ' ') ?? '';
  }
}

// TODO: Load this up with some .csv data
function createLargeDataFrame() {
  const a = Series.new(Array.from(Array(300).keys()));
  return new DataFrame({'a': a, 'b': a});
}
