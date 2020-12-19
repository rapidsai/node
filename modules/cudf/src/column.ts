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
import { MemoryData } from '@nvidia/cuda';
import { DataType, TypeId } from './types';
import { DeviceBuffer } from '@nvidia/rmm';

interface ColumnConstructor {
  readonly prototype: Column;
  new (props: {
    type: DataType | TypeId;
    data?: DeviceBuffer | MemoryData | null;
    offset?: number;
    length?: number;
    nullCount?: number;
    nullMask?: DeviceBuffer | MemoryData | null;
    children?: ReadonlyArray<Column> | null;
  }): Column;
}

export interface Column {
  [index: number]: any;

  readonly type: DataType;
  readonly data: DeviceBuffer;
  readonly mask: DeviceBuffer;

  readonly length: number;
  readonly nullable: boolean;
  readonly hasNulls: boolean;
  readonly nullCount: number;
  readonly numChildren: number;

  getChild(index: number): Column;

  getValue(index: number): this[0];
  // setValue(index: number, value?: this[0] | null): void;

  setNullCount(nullCount: number): void;
  setNullMask(mask: DeviceBuffer, nullCount?: number): void;
}

// eslint-disable-next-line @typescript-eslint/no-redeclare
export const Column: ColumnConstructor = CUDF.Column;

Object.setPrototypeOf(
  CUDF.Column.prototype,
  new Proxy(
    {},
    {
      get(target: any, p: any, column: any) {
        let i: number = p;
        switch (typeof p) {
          // @ts-ignore
          case 'string':
            if (isNaN((i = +p))) {
              break;
            }
          // eslint-disable-next-line no-fallthrough
          case 'number':
            if (i > -1 && i < column.length) {
              return column.getValue(i);
            }
            return undefined;
        }
        return Reflect.get(target, p, column);
      },
    },
  ),
);
