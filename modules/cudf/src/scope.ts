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

import {DataFrame} from './data_frame';
import {Series} from './series';

let scopeID = 0;

export function scope<T extends DataFrame|Series, F extends(() => T | Promise<T>)>(cb: F):
  ReturnType<F> {
  const resources = [] as (DataFrame<any>| Series<any>)[];
  const new_id    = scopeID++;
  const old_id    = DataFrame.scopeID;

  DataFrame.scopeID             = new_id;
  DataFrame.disposables[new_id] = resources;

  Series.scopeID             = new_id;
  Series.disposables[new_id] = resources;

  function cleanup(value: any) {
    for (const resource of resources) {
      if (resource !== value) { resource.dispose(); }
    }
    delete DataFrame.disposables[new_id];
    delete Series.disposables[new_id];
  }

  const result = cb();

  DataFrame.scopeID = old_id;
  Series.scopeID    = old_id;

  if (result instanceof Promise) {
    // eslint-disable-next-line @typescript-eslint/no-floating-promises
    result.then(cleanup);
  } else {
    cleanup(result);
  }

  return result as any;
}
