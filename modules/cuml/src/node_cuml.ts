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

import {DataType} from '@rapidsai/cudf';
import {DeviceBuffer} from '@rapidsai/rmm';

import {COOConstructor} from './coo';
import {UMAPConstructor} from './umap_base';

/** @ignore */
export declare const _cpp_exports: any;

export declare const COO: COOConstructor;

export declare const UMAP: UMAPConstructor;

export declare function trustworthiness(features: any[]|DeviceBuffer,
                                        featuresType: DataType,
                                        embedding: any[]|DeviceBuffer,
                                        embeddedType: DataType,
                                        nSamples: number,
                                        nFeatures: number,
                                        nComponents: number,
                                        nNeighbors: number,
                                        batch_size: number): number;
