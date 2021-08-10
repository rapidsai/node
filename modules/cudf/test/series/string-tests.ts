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
import {
  Float32,
  Float64,
  Int16,
  Int32,
  Int64,
  Int8,
  Series,
  StringSeries,
  Uint16,
  Uint32,
  Uint64,
  Uint8
} from '@rapidsai/cudf';
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

describe('StringSeries', () => {
  test('Can concat', () => {
    const col         = StringSeries.new(['foo']);
    const colToConcat = StringSeries.new(['bar']);

    const result = col.concat(colToConcat);

    expect([...result]).toEqual([...col, ...colToConcat]);
  });
});

describe('StringSeries', () => {
  test('Can copy', () => {
    const col = StringSeries.new(['foo', 'bar']);

    const result = col.copy();

    expect([...result]).toEqual([...col]);
  });
});

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
  const a = Series.new(object_data.map((x) => JSON.stringify(x)));

  // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
  expect(JSON.parse(a.getJSONObject('$.goat').getValue(0)!)).toEqual(object_data[0].goat);
  // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
  expect(JSON.parse(a.getJSONObject('$.leopard').getValue(1)!)).toEqual(object_data[1].leopard);

  const b = Series.new(['']);
  expect([...b.getJSONObject('$')]).toStrictEqual([null]);
  expect([...b.getJSONObject('')]).toStrictEqual([]);
});

function testIntegralCast<T extends Int8|Int16|Int32|Uint8|Uint16|Uint32>(type: T) {
  const a = Series.new(['0', '1', '2', null]);
  expect([...a.cast(type)]).toStrictEqual([0, 1, 2, null]);
}

function testBigIntegralCast<T extends Uint64|Int64>(type: T) {
  const a = Series.new(['0', '1', '2', null]);
  expect([...a.cast(type)]).toStrictEqual([0n, 1n, 2n, null]);
}

describe('Series.cast Integral', () => {
  test('Int8', () => { testIntegralCast(new Int8); });
  test('Int16', () => { testIntegralCast(new Int16); });
  test('Int32', () => { testIntegralCast(new Int32); });
  test('Int64', () => { testBigIntegralCast(new Int64); });
  test('Uint8', () => { testIntegralCast(new Uint8); });
  test('Uint16', () => { testIntegralCast(new Uint16); });
  test('Uint32', () => { testIntegralCast(new Uint32); });
  test('Uint64', () => { testBigIntegralCast(new Uint64); });
});

test('Series.cast Float32', () => {
  const a = Series.new(['0', '2.5', '-2', '10.2', null, '2.48e+2']);
  expect([...a.cast(new Float32)]).toStrictEqual([0, 2.5, -2, 10.199999809265137, null, 248.0]);
});

test('Series.cast Float64', () => {
  const a = Series.new(['0', '2.5', '-2', '10.2', null, '2.48e+2']);
  expect([...a.cast(new Float64)]).toStrictEqual([0, 2.5, -2, 10.2, null, 248.0]);
});
