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

/* eslint-disable @typescript-eslint/no-redeclare */

import {addon as CUDA} from '@rapidsai/cuda';
import {addon as CUDF} from '@rapidsai/cudf';
import {addon as RMM} from '@rapidsai/rmm';

export const {
  parseSchema,
  getTableScanInfo,
  runGeneratePhysicalGraph,
  Context,
  UcpContext,
  ExecutionGraph
} = require('bindings')('rapidsai_sql.node').init(CUDA, RMM, CUDF) as
    typeof import('./rapidsai_sql') ;

export type getTableScanInfo         = typeof import('./rapidsai_sql').getTableScanInfo;
export type runGeneratePhysicalGraph = typeof import('./rapidsai_sql').runGeneratePhysicalGraph;
export type parseSchema              = typeof import('./rapidsai_sql').parseSchema;

export type Context        = import('./rapidsai_sql').Context;
export type UcpContext     = import('./rapidsai_sql').UcpContext;
export type ExecutionGraph = import('./rapidsai_sql').ExecutionGraph;
export type ContextProps   = import('./rapidsai_sql').ContextProps;
export type WorkerUcpInfo  = import('./rapidsai_sql').WorkerUcpInfo;
