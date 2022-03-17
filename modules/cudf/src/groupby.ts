// Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

import {MemoryResource} from '@rapidsai/rmm';

import * as CUDF from './addon';
import {Column} from './column';
import {Table} from './table';
import {NullOrder} from './types/enums';

export {GroupByMultiple, GroupByMultipleProps} from './groupby/multiple';
export {GroupBySingle, GroupBySingleProps} from './groupby/single';

export interface GroupByBaseProps {
  include_nulls?: boolean;
  keys_are_sorted?: boolean;
  column_order?: boolean[];
  null_precedence?: NullOrder[];
}

export interface GroupByProps extends GroupByBaseProps {
  keys: Table;
}

export interface GroupByConstructor {
  readonly prototype: GroupBy;
  new(props: GroupByProps): GroupBy;
}

export interface GroupBy {
  _getGroups(values?: Table,
             memoryResource?: MemoryResource):  //
    {keys: Table, offsets: Int32Array, values?: Table};

  _argmax(values: Table, memoryResource?: MemoryResource):  //
    {keys: Table, cols: Column[]};

  _argmin(values: Table, memoryResource?: MemoryResource):  //
    {keys: Table, cols: Column[]};

  _count(values: Table, memoryResource?: MemoryResource):  //
    {keys: Table, cols: Column[]};

  _max(values: Table, memoryResource?: MemoryResource):  //
    {keys: Table, cols: Column[]};

  _mean(values: Table, memoryResource?: MemoryResource):  //
    {keys: Table, cols: Column[]};

  _median(values: Table, memoryResource?: MemoryResource):  //
    {keys: Table, cols: Column[]};

  _min(values: Table, memoryResource?: MemoryResource):  //
    {keys: Table, cols: Column[]};

  _nth(values: Table, memoryResource?: MemoryResource, n?: number, include_nulls?: boolean):
    {keys: Table, cols: Column[]};

  _nunique(values: Table, memoryResource?: MemoryResource, include_nulls?: boolean):
    {keys: Table, cols: Column[]};

  _std(values: Table, memoryResource?: MemoryResource, ddof?: number):
    {keys: Table, cols: Column[]};

  _sum(values: Table, memoryResource?: MemoryResource):  //
    {keys: Table, cols: Column[]};

  _var(values: Table, memoryResource?: MemoryResource, ddof?: number):
    {keys: Table, cols: Column[]};

  _quantile(values: Table, memoryResource?: MemoryResource, q?: number, interpolation?: number):
    {keys: Table, cols: [Column]};

  _collect_list(values: Table, memoryResource?: MemoryResource, include_nulls?: boolean):
    {keys: Table, cols: Column[]};

  _collect_set(values: Table,
               memoryResource?: MemoryResource,
               include_nulls?: boolean,
               nulls_equal?: boolean,
               nans_equal?: boolean):  //
    {keys: Table, cols: Column[]};
}

// eslint-disable-next-line @typescript-eslint/no-redeclare
export const GroupBy: GroupByConstructor = CUDF.GroupBy;
