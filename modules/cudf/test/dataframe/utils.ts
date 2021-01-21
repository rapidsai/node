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

import {toArray} from 'ix/asynciterable';

export function makeCSVString(
  opts: {rows?: any[], delimiter?: string, lineTerminator?: string, header?: boolean} = {}) {
  const {rows = [], delimiter = ',', lineTerminator = '\n', header = true} = opts;
  const names = Object.keys(rows.reduce(
    (keys, row) => Object.keys(row).reduce((keys, key) => ({...keys, [key]: true}), keys), {}));
  return [
    ...[header ? names.join(delimiter) : []],
    ...rows.map((row) =>
                  names.map((name) => row[name] === undefined ? '' : row[name]).join(delimiter))
  ].join(lineTerminator) +
         lineTerminator;
}

export async function toStringAsync(source: AsyncIterable<string>) {
  return (await toArray(source)).join('');
}
