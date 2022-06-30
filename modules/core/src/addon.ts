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

let addon = {
  _cpp_exports: null as any,
  getComputeCapabilities() { return new Array<string>(); },
};

try {
  addon = require('bindings')('rapidsai_core.node').init() as typeof addon;
} catch {  //
  /**/
}

// eslint-disable-next-line @typescript-eslint/unbound-method
export const {_cpp_exports, getComputeCapabilities} = addon;
