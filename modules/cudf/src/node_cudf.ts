// Copyright (c) 2022, NVIDIA CORPORATION.
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

import {ColumnConstructor} from './column';
import {GroupByConstructor} from './groupby';
import {ScalarConstructor} from './scalar';
import {TableConstructor} from './table';
import {DataType} from './types/dtypes';
import {CommonType} from './types/mappings';

/** @ignore */
export declare const _cpp_exports: any;

export declare const Column: ColumnConstructor;
export declare const GroupBy: GroupByConstructor;
export declare const Scalar: ScalarConstructor;
export declare const Table: TableConstructor;

export declare function findCommonType<T extends DataType, R extends DataType>(
  lhs: T, rhs: R): CommonType<T, R>;
