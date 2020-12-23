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

import * as Path from 'path';

const NODE_DEBUG = ((<any>process.env).NODE_DEBUG || (<any>process.env).NODE_ENV === 'debug');

export function loadNativeModule<T = any>({id}: import('module'), name: string): T {
  let moduleBasePath            = Path.dirname(id);
  let nativeModule: T|undefined = undefined;
  const errors                  = [`\nFailed to load "${name}:"\n`];
  // HACK: Adjust base path if running in Jest.
  // TODO: Figure out how to use Jest's moduleNameMapper config
  // even though this file is in a module Jest doesn't compile.
  if (Path.basename(moduleBasePath) == 'src') {
    moduleBasePath = Path.dirname(moduleBasePath);
    moduleBasePath = Path.join(moduleBasePath, 'build', 'js');
  }
  for (const type of (NODE_DEBUG ? ['Debug', 'Release'] : ['Release'])) {
    try {
      if ((nativeModule = require(Path.join(moduleBasePath, '..', type, `${name}.node`)))) {
        break;
      }
    } catch (e) {
      errors.push(e);
      continue;
    }
  }
  if (nativeModule) {
    if (typeof (<any>nativeModule).init === 'function') {
      return (<any>nativeModule).init() || nativeModule;
    }
    return nativeModule;
  }
  throw new Error(errors.join('\n'));
}
