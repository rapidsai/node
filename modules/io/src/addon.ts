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

/* eslint-disable @typescript-eslint/no-redeclare */

// import {loadNativeModule} from '@rapidsai/core';
// import {addon as CUDA} from '@rapidsai/cuda';
// import {addon as CUDF} from '@rapidsai/cudf';
// import {addon as RMM} from '@rapidsai/rmm';

// export const {} = loadNativeModule<typeof import('./rapidsai_io')>(
//   module, 'rapidsai_io', init => init(CUDA, RMM, CUDF));

export type testMethod = typeof import('./rapidsai_io').testMethod;
