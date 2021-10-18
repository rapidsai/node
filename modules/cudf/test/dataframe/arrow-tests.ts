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

import {DataFrame} from '@rapidsai/cudf';
import {FloatVector, IntVector, Table} from 'apache-arrow';
import {ChildProcessByStdio, spawn} from 'child_process';
import {Readable, Writable} from 'stream';

jest.setTimeout(60 * 1000);

test(`fromArrow works from host memory`, () => {
  const table = Table.new(
    [
      FloatVector.from(new Float64Array([1.1, 2.2, 0, -3.3, -4.4])),
      IntVector.from(new Int32Array([1, 2, 0, -3, -4]))
    ],
    ['floats', 'ints']);
  const serialized_table = table.serialize();  // Uint8Array
  const df               = DataFrame.fromArrow(serialized_table);

  expect([...df.names]).toStrictEqual(['floats', 'ints']);
  expect([...df.get('floats')]).toStrictEqual([1.1, 2.2, 0, -3.3, -4.4]);
  expect([...df.get('ints')]).toStrictEqual([1, 2, 0, -3, -4]);
});

test(`fromArrow works between subprocesses`, async () => {
  let src: ChildProcessByStdio<Writable, Readable, null>|undefined;
  let dst: ChildProcessByStdio<Writable, Readable, null>|undefined;
  try {
    src          = spawnIPCSourceSubprocess();
    const handle = await readChildProcessOutput(src);
    if (handle) {
      dst        = spawnIPCTargetSubprocess(JSON.parse(handle));
      const data = await readChildProcessOutput(dst);
      if (data) {
        expect(JSON.parse(data))
          .toStrictEqual(
            {names: ['floats', 'ints'], a: [1.1, 2.2, 0, -3.3, -4.4], b: [1, 2, 0, -3, -4]});
      } else {
        throw new Error(`Invalid data from target child process: ${JSON.stringify(data)}`);
      }
    } else {
      throw new Error(`Invalid IPC handle from source child process: ${JSON.stringify(handle)}`);
    }
  } finally {
    dst && !dst.killed && dst.kill();
    src && !src.killed && src.kill();
  }
});

async function readChildProcessOutput(proc: ChildProcessByStdio<Writable, Readable, null>) {
  const {stdout} = proc;
  return (async () => {
    for await (const chunk of stdout) {
      if (chunk) {
        // eslint-disable-next-line @typescript-eslint/restrict-plus-operands
        return '' + chunk;
      }
    }
    return '';
  })();
}

function spawnIPCSourceSubprocess() {
  return spawn('node',
               [
                 `-e`,
                 `
const { Table, FloatVector, IntVector } = require('apache-arrow');
const { Uint8Buffer } = require('@rapidsai/cuda');
const table = Table.new(
  [
    FloatVector.from(new Float64Array([1.1, 2.2, 0, -3.3, -4.4])),
    IntVector.from(new Int32Array([1, 2, 0, -3, -4]))
  ],
  ['floats', 'ints']
);
const serialized_table = table.serialize();  // Uint8Array
const device_buffer = new Uint8Buffer(serialized_table.length);
device_buffer.copyFrom(serialized_table);
const handle = device_buffer.getIpcHandle();

process.stdout.write(JSON.stringify(handle));
process.on("exit", () => handle.close());
setInterval(() => { }, 60 * 1000);
`
               ],
               {stdio: ['pipe', 'pipe', 'inherit']});
}

function spawnIPCTargetSubprocess({handle}: {handle: Array<number>}) {
  return spawn('node',
               [
                 '-e',
                 `
const { IpcMemory } = require("@rapidsai/cuda");
const { DataFrame } = require(".");
const dmem = new IpcMemory([${handle.toString()}]);
const df = DataFrame.fromArrow(dmem);

process.stdout.write(JSON.stringify({ names: [...df.names], a: [...df.get('floats')], b: [...df.get('ints')] }));
`
               ],
               {stdio: ['pipe', 'pipe', 'inherit']});
}
