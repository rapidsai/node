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

import {DeviceBuffer} from '@rapidsai/rmm';

import {Column} from './column';
import {DataFrame} from './data_frame';
import {Series} from './series';
import {Table} from './table';

type Resource = DataFrame|Series|Column|Table|DeviceBuffer;

export class Disposer {
  private currentScopeId                 = -1;
  private trackedResources: Resource[][] = [];
  private ingoredResources: Resource[][] = [];

  add(value: Resource) {
    const {currentScopeId} = this;
    if (currentScopeId > -1) { this.trackedResources[currentScopeId].push(value); }
  }

  enter(doNotDispose: Resource[]) {
    this.ingoredResources.push(doNotDispose);
    this.currentScopeId = this.trackedResources.push([]) - 1;
    return this;
  }

  exit(result: any) {
    if (this.currentScopeId > -1) {
      this.currentScopeId -= 1;
      const flatten = (xs: Resource[]) => flattenDeviceBuffers(xs);
      // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
      const test = new Set(flatten(this.trackedResources.pop()!));
      const keep = new Set(flatten([
        result,
        // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
        ...this.ingoredResources.pop()!,
        ...this.ingoredResources.flat(1),
        ...this.trackedResources.flat(1),
      ]));

      for (const buf of test) {
        if (!keep.has(buf)) { buf.dispose(); }
      }
    }
    return result;
  }
}

export const DISPOSER = new Disposer();

export function scope<T extends Resource, F extends(() => T | Promise<T>)>(
  cb: F, doNotDispose: Resource[] = []) {
  DISPOSER.enter(doNotDispose);
  const result = cb();
  if (result instanceof Promise) {  //
    return result.then((x) => DISPOSER.exit(x)) as ReturnType<F>;
  }
  return DISPOSER.exit(result) as ReturnType<F>;
}

function flattenDeviceBuffers(input: Resource|null|undefined, visited = new Set): DeviceBuffer[] {
  if (!input) { return []; }
  if (input instanceof DeviceBuffer) { return [input]; }
  if (input instanceof Series) { return flattenDeviceBuffers(input._col, visited); }
  if (input instanceof DataFrame) { return flattenDeviceBuffers(input.asTable(), visited); }
  if (Array.isArray(input)) { return input.flatMap((x) => flattenDeviceBuffers(x, visited)); }
  if (input instanceof Column) {
    let bufs = [input.data, input.mask];
    for (let i = -1; ++i < input.numChildren;) {
      bufs = bufs.concat(flattenDeviceBuffers(input.getChild(i), visited));
    }
    return bufs;
  }
  if (input instanceof Table) {
    let bufs: DeviceBuffer[] = [];
    for (let i = -1; ++i < input.numColumns;) {
      bufs = bufs.concat(flattenDeviceBuffers(input.getColumnByIndex(i), visited));
    }
    return bufs;
  }
  if (typeof input === 'object' && !visited.has(input)) {
    return flattenDeviceBuffers(Object.values(input), visited);
  }
  return [];
}
