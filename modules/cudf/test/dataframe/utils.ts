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

export function makeCSVString(
  opts: {rows?: any[], delimitor?: string, lineTerminator?: string, header?: boolean} = {}) {
  const {rows = [], delimitor = ',', lineTerminator = '\n', header = true} = opts;
  const names = Object.keys(rows.reduce(
    (keys, row) => Object.keys(row).reduce((keys, key) => ({...keys, [key]: true}), keys), {}));
  return [
    ...(function*() {
      if (header) yield names.join(delimitor);
      for (const row of rows) {
        yield names.map((name) => row[name] === undefined ? '' : row[name]).join(delimitor);
      }
    })()
  ].join(lineTerminator) +
         lineTerminator;
}

export async function streamToString(stream: NodeJS.ReadableStream) {
  let str = '';
  for await (const chunk of stream[Symbol.asyncIterator]()) {  //
    str += chunk.toString();
  }
  return str;
}
