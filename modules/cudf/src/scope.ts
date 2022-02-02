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

export class Disposer {
  private counter                                            = 0;
  private id: number|null                                            = null;
  private resources: {[key: number]: (Series<any>|DataFrame<any>)[]} = {};

  add(value: DataFrame|Series) {
    if (this.id != null) { this.resources[this.id].push(value); }
  }

  enter(): [(value: any) => void, number|null] {
    const old_id = this.id;
    this.id      = this.counter++;

    const ID        = this.id;
    const resources = this.resources;
    resources[ID]   = [];

    const cleanup = (value: any) => {
      for (const resource of resources[ID]) {
        if (resource !== value) { resource.dispose(); }
      }
      delete resources[ID];
    };
    return [cleanup, old_id];
  }

  exit(old_id: number|null) { this.id = old_id; }
}

export const DISPOSER = new Disposer();

export function scope<T extends DataFrame|Series, F extends(() => T | Promise<T>)>(cb: F):
  ReturnType<F> {
  const [cleanup, old_id] = DISPOSER.enter();
  const result            = cb();
  DISPOSER.exit(old_id);

  if (result instanceof Promise) {
    // eslint-disable-next-line @typescript-eslint/no-floating-promises
    result.then(cleanup);
  } else {
    cleanup(result);
  }

  return result as any;
}
