// Copyright (c) 2020, NVIDIA CORPORATION.
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

import CUDF from './addon';
import {Column} from './column';
import {DataType, NullOrder} from './types';

type ToArrowMetadata = [string | number, ToArrowMetadata?];

interface TableConstructor {
  readonly prototype: Table;
  new(props: {columns?: ReadonlyArray<Column>|null}): Table;
}

export interface Table {
  readonly numColumns: number;
  readonly numRows: number;
  getColumnByIndex<T extends DataType = any>(index: number): Column<T>;
<<<<<<< HEAD
  toArrow(names: ToArrowMetadata[]): Uint8Array;
=======
  orderBy(column_orders: boolean[], null_orders: NullOrder[]): Column;
>>>>>>> 8d7457d (checkpoint)
}

// eslint-disable-next-line @typescript-eslint/no-redeclare
export const Table: TableConstructor = CUDF.Table;
