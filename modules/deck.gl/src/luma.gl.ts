// Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

import {
  Buffer as LumaBuffer_,
} from '@luma.gl/webgl';

// Add protected props and methods missing on the luma.gl typings
export declare class LumaResource {
  _deleteHandle(handle?: any): void;
}

export declare class LumaBuffer extends LumaBuffer_ {
  // _handle isn't mutable in practice, even though it is in the luma.gl typings
  _handle: any;
  _deleteHandle(): void;
  _setData(data: any, offset?: number, byteLength?: number): void;
  _setByteLength(byteLength: number, usage?: number): void;
}
