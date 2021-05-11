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

import {setDefaultAllocator} from '@nvidia/cuda';
import {Series, StringSeries} from '@rapidsai/cudf';
import {CudaMemoryResource, DeviceBuffer} from '@rapidsai/rmm';

const mr = new CudaMemoryResource();

setDefaultAllocator((byteLength: number) => new DeviceBuffer(byteLength, mr));

const data: string[] = [
  'foo bar baz',   // start of string
  ' foo bar baz',  // start of string after whitespace
  'baz bar foo',   // end of string
  'foo bar foo',   // start and end
  'bar foo baz',   // middle
  'baz quux',      // missing
  'FoO',           // wrong case
  'bar\n foo',     // multi-line
  'bar\nfoo'       // multi-line
];

describe.each([['foo'], [/foo/], [/foo/ig]])('Series regex search (pattern=%p)', (pattern) => {
  test('containsRe', () => {
    const expected = [true, true, true, true, true, false, false, true, true];
    const s        = Series.new(data);
    expect([...s.containsRe(pattern)]).toEqual(expected);
  });

  test('countRe', () => {
    const expected = [1, 1, 1, 2, 1, 0, 0, 1, 1];
    const s        = StringSeries.new(data);
    expect([...s.countRe(pattern)]).toEqual(expected);
  });

  test('matchesRe', () => {
    const expected = [true, false, false, true, false, false, false, false, false];
    const s        = StringSeries.new(data);
    expect([...s.matchesRe(pattern)]).toEqual(expected);
  });
});

// getJSONObject tests
test('getJSONObject', () => {
  const object_data =
    [{goat: {id: 0, species: 'Capra Hircus'}}, {leopard: {id: 1, species: 'Panthera pardus'}}];
  const a = Series.new((object_data as any).map(JSON.stringify));

  expect(JSON.parse(a.getJSONObject('$.goat').getValue(0))).toEqual(object_data[0].goat);
  expect(JSON.parse(a.getJSONObject('$.leopard').getValue(1))).toEqual(object_data[1].leopard);

  const b = Series.new(['']);
  expect([...b.getJSONObject('$')]).toStrictEqual([null]);
  expect([...b.getJSONObject('')]).toStrictEqual([]);
});
