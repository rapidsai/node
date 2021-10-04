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

import {ContextProps, UcpContext} from '../addon';
import {SQLContext} from '../context';

let context: SQLContext;

function die({code = 0}: any) { process.exit(code); }

// eslint-disable-next-line @typescript-eslint/no-unused-vars
function init({uuid, ...props}: {uuid: string}&ContextProps) {
  context = new SQLContext({...props, ucpContext: new UcpContext()});
}

function dropTable({name}: {name: string}) { context.dropTable(name); }

async function createTable({name, table_id}: {name: string, table_id: string}) {
  context.createTable(name, await context.pull(table_id));
}

function createCSVTable({name, path}: {name: string, path: string}) {
  context.createTable(name, [path]);
}

async function sql({uuid, query, token}: {uuid: string, query: string, token: number}) {
  await context.sql(query, token).sendTo(0, uuid);
}

process.on('message', ({type, ...opts}: any) => {
  // eslint-disable-next-line @typescript-eslint/no-floating-promises
  (async () => {
    switch (type) {
      case 'kill': return die(opts);
      case 'init': return init(opts);
      case 'sql': return await sql(opts);
      case 'dropTable': return dropTable(opts);
      case 'createTable': return await createTable(opts);
      case 'createCSVTable': return createCSVTable(opts);
    }
    return {};
  })()
    .catch((error) => {
      if (opts.uuid && process.send) { process.send({error, uuid: opts.uuid}); }
    })
    .then((res: any) => {
      if (opts.uuid && process.send) { process.send({...res, uuid: opts.uuid}); }
    });
});
