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

import {Column} from './column';
import {DataFrame} from './data_frame';
import {Series} from './series';
import {Table} from './table';

type Resource = DataFrame|Series|Column|Table;

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
      // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
      const test = new Set(this.trackedResources.pop()!.flatMap(flattenColumns));
      const keep = new Set([
        result,
        // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
        ...this.ingoredResources.pop()!,
        ...this.ingoredResources.flat(1),
        ...this.trackedResources.flat(1),
      ].flatMap(flattenColumns));
      for (const col of test) {
        if (!keep.has(col)) { col.dispose(); }
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

function flattenColumns(input?: Resource): Column[] {
  if (!input) { return []; }
  if (Array.isArray(input)) { return input.flatMap(flattenColumns); }
  if (input instanceof Series) { return flattenColumns(input._col); }
  if (input instanceof DataFrame) { return flattenColumns(input.asTable()); }
  let cols: Column<any>[] = [];
  if (input instanceof Column) {
    cols = [input];
    for (let i = -1; ++i < input.numChildren;) {
      cols = cols.concat(flattenColumns(input.getChild(i)));
    }
  } else if (input instanceof Table) {
    for (let i = -1; ++i < input.numColumns;) {
      cols = cols.concat(flattenColumns(input.getColumnByIndex(i)));
    }
  }
  return cols;
}
