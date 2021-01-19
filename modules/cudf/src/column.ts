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

import {MemoryData} from '@nvidia/cuda';
import {DeviceBuffer} from '@nvidia/rmm';

import CUDF from './addon';
import {DataType, TypeId} from './types';

export type ColumnProps = {
  // todo -- need to pass full DataType instance when we implement fixed_point
  type: TypeId,
  data?: DeviceBuffer|MemoryData|null,
  offset?: number,
  length?: number,
  nullCount?: number,
  nullMask?: DeviceBuffer|MemoryData|null,
  children?: ReadonlyArray<Column>|null
};

interface ColumnConstructor {
  readonly prototype: Column;
  new<T extends DataType = any>(props: ColumnProps): Column<T>;
}

/**
 * A low-level wrapper for libcudf Column Objects
 */
export interface Column<T extends DataType = any> {
  readonly type: T;
  readonly data: DeviceBuffer;
  readonly mask: DeviceBuffer;

  readonly length: number;
  readonly nullable: boolean;
  readonly hasNulls: boolean;
  readonly nullCount: number;
  readonly numChildren: number;

  /**
   * Return a child at the specified index to host memory
   *
   * @param index
   */
  getChild(index: number): Column;

  /**
   * Return a value at the specified index to host memory
   *
   * @param index
   */
  getValue(index: number): T['valueType']|null;

  // setValue(index: number, value?: T['valueType'] | null): void;

  /**
   * Set the null count for the null mask
   *
   * @param nullCount
   */
  setNullCount(nullCount: number): void;

  /**
   *
   * @param mask The null-mask. Valid values are marked as 1; otherwise 0. The
   * mask bit given the data index idx is computed as:
   * ```
   * (mask[idx // 8] >> (idx % 8)) & 1
   * ```
   * @param nullCount The number of null values. If None, it is calculated
   * automatically.
   */
  setNullMask(mask: DeviceBuffer, nullCount?: number): void;
}

// eslint-disable-next-line @typescript-eslint/no-redeclare
export const Column: ColumnConstructor = CUDF.Column;
