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

import {DataFrame} from '@rapidsai/cudf';

import {UcpContext} from './addon';
import {
  CONFIG_OPTIONS,
  CREATE_BLAZING_CONTEXT,
  CREATE_TABLE,
  QUERY_RAN,
  RUN_QUERY
} from './blazingcluster';
import {BlazingContext} from './blazingcontext';

let bc: BlazingContext;

process.on('message', (args: Record<string, unknown>) => {
  const {operation, ...rest} = args;

  if (operation == CREATE_BLAZING_CONTEXT) {
    const ralId              = rest['ralId'] as number;
    const ucpMetaData: any[] = rest['ucpMetadata'] as Record<string, any>[];
    const ucpContext         = new UcpContext();

    /* eslint-disable @typescript-eslint/no-unused-vars */
    // @ts-ignore
    bc = new BlazingContext({
      ralId: ralId,
      ralCommunicationPort: 4000 + ralId,
      configOptions: {...CONFIG_OPTIONS},
      workersUcpInfo: ucpMetaData.map((xs: any) => ({...xs, ucpContext}))
    });

    console.log(`created context: ${ralId}`);

  } else if (operation == CREATE_TABLE) {
    const tableName = rest['tableName'] as string;
    const dataframe = rest['dataframe'] as any;

    bc.createTable(tableName, DataFrame.fromArrow(dataframe));

    console.log(`created table: ${tableName}`);
    console.log(DataFrame.fromArrow(dataframe));
  } else if (operation == RUN_QUERY) {
    const query     = rest['query'] as string;
    const ctxToken  = rest['ctxToken'] as number;
    const messageId = rest['messageId'] as string;

    bc.sql(query, ctxToken).sendTo(0, messageId);
    (<any>process).send({operation: QUERY_RAN, ctxToken, messageId});
  }
});
