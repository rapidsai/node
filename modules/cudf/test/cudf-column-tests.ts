// Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

import {
  DeviceMemory,
  Float32Buffer,
  Int32Buffer,
  setDefaultAllocator,
  Uint8Buffer
} from '@rapidsai/cuda';
import {
  Bool8,
  Column,
  DataFrame,
  DataType,
  Float32,
  Float64,
  Int32,
  Series,
  SeriesMap,
  StringSeries,
  Uint8,
  Utf8String
} from '@rapidsai/cudf';
import {CudaMemoryResource, DeviceBuffer} from '@rapidsai/rmm';
import {BoolVector} from 'apache-arrow';
import {promises} from 'fs';
import * as Path from 'path';

/* TODO: How do I apply a list of dtypes?
 */
function json_aos_to_dataframe(
  str: StringSeries, columns: ReadonlyArray<string>, _: ReadonlyArray<DataType>): DataFrame {
  const arr = {} as SeriesMap;
  columns.forEach((col, ix) => {
    const no_open_list = str.split('[\n').gather([1], false);
    const tokenized    = no_open_list.split('},');
    console.log(tokenized.toArray());
    const parse_result = tokenized._col.getJSONObject('.' + columns[ix]);
    arr[col]           = Series.new(parse_result);
    console.log(Series.new(parse_result).toArray());
  });
  const result = new DataFrame(arr);
  return result;
}
/* TODO: How do I apply a list of dtypes?
 */
function json_aoa_to_dataframe(str: StringSeries, dtypes: ReadonlyArray<DataType>): DataFrame {
  const arr = {} as SeriesMap;
  dtypes.forEach((_, ix) => {
    const no_open_list = str.split('[\n').gather([1], false);
    const tokenized    = no_open_list.split('],');
    const get_ix       = `[${ix}]`;
    const parse_result = tokenized._col.getJSONObject(get_ix);
    arr[ix]            = Series.new(parse_result);
  });
  const result = new DataFrame(arr);
  return result;
}

describe('Graphology dataset parsing', () => {
  test('extracts four objects from the base object', () => {
    const dataset   = StringSeries.read_text('dataset_small.json.txt', '');
    let split       = dataset.split('"tags":');
    const ttags     = split.gather([1], false);
    let rest        = split.gather([0], false);
    split           = rest.split('"clusters":');
    const tclusters = split.gather([1], false);
    rest            = split.gather([0], false);
    split           = rest.split('"edges":');
    const tedges    = split.gather([1], false);
    rest            = split.gather([0], false);
    split           = rest.split('"nodes":');
    const tnodes    = split.gather([1], false);
    const tags = json_aos_to_dataframe(ttags, ['key', 'image'], [new Utf8String, new Utf8String]);
    const clusters = json_aos_to_dataframe(
      tclusters, ['key', 'color', 'clusterLabel'], [new Int32, new Utf8String, new Utf8String]);
    const nodes =
      json_aos_to_dataframe(tnodes, ['key', 'label', 'tag', 'URL', 'cluster', 'x', 'y', 'score'], [
        new Utf8String,
        new Utf8String,
        new Utf8String,
        new Utf8String,
        new Int32,
        new Float64,
        new Float64,
        new Int32
      ]);
    const edges = json_aoa_to_dataframe(tedges, [new Utf8String, new Utf8String]);
    expect(nodes.names).toEqual(['key', 'label', 'tag', 'URL', 'cluster', 'x', 'y', 'score']);
    expect(nodes.numRows).toEqual(5);
    expect(edges.numRows).toEqual(11);
    expect(clusters.names).toEqual(['key', 'color', 'clusterLabel']);
    expect(clusters.numRows).toEqual(24);
    expect(tags.names).toEqual(['key', 'image']);
    expect(tags.numRows).toEqual(11);
  });
});

describe('Column.read_text', () => {
  test('can read a json file', async () => {
    const rows = [
      {a: 0, b: 1.0, c: '2'},
      {a: 1, b: 2.0, c: '3'},
      {a: 2, b: 3.0, c: '4'},
    ];
    const outputString = JSON.stringify(rows);
    const path         = Path.join(readTextTmpDir, 'simple.txt');
    await promises.writeFile(path, outputString);
    const text = StringSeries.read_text(path, '');
    expect(text.getValue(0)).toEqual(outputString);
    await new Promise<void>((resolve, reject) =>
                              rimraf(path, (err?: Error|null) => err ? reject(err) : resolve()));
  });
  test('can read a random file', async () => {
    const outputString = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()';
    const path         = Path.join(readTextTmpDir, 'simple.txt');
    await promises.writeFile(path, outputString);
    const text = StringSeries.read_text(path, '');
    expect(text.getValue(0)).toEqual(outputString);
    await new Promise<void>((resolve, reject) =>
                              rimraf(path, (err?: Error|null) => err ? reject(err) : resolve()));
  });
  test('can read an empty file', async () => {
    const outputString = '';
    const path         = Path.join(readTextTmpDir, 'simple.txt');
    await promises.writeFile(path, outputString);
    const text = StringSeries.read_text(path, '');
    expect(text.getValue(0)).toEqual(outputString);
    await new Promise<void>((resolve, reject) =>
                              rimraf(path, (err?: Error|null) => err ? reject(err) : resolve()));
  });
});

let readTextTmpDir = '';

const rimraf = require('rimraf');

beforeAll(async () => {  //
  readTextTmpDir = await promises.mkdtemp(Path.join('/tmp', 'node_cudf'));
});

afterAll(() => {
  return new Promise<void>((resolve, reject) => {  //
    rimraf(readTextTmpDir, (err?: Error|null) => err ? reject(err) : resolve());
  });
});
const mr = new CudaMemoryResource();

setDefaultAllocator((byteLength) => new DeviceBuffer(byteLength, mr));

describe('Column split', () => {
  test('split a basic string', () => {
    const input   = Series.new(['abcdefg']);
    const example = Series.new(['abcd', 'efg']);
    const result  = Series.new(input._col.split('d'));
    expect(result).toEqual(example);
  });
  test('split a string twice', () => {
    const input   = Series.new(['abcdefgdcba']);
    const example = Series.new(['abcd', 'efgd', 'cba']);
    const result  = Series.new(input._col.split('d'));
    expect(result).toEqual(example);
  });
});

test('Column initialization', () => {
  const length = 100;
  const col    = new Column({type: new Int32, data: new Int32Buffer(length)});

  expect(col.type).toBeInstanceOf(Int32);
  expect(col.length).toBe(length);
  expect(col.nullCount).toBe(0);
  expect(col.hasNulls).toBe(false);
  expect(col.nullable).toBe(false);
});

test('Column initialization with null_mask', () => {
  const length = 100;
  const col    = new Column({
    type: new Bool8,
    data: new Uint8Buffer(length),
    nullMask: new Uint8Buffer(64).fill(0),
  });

  expect(col.length).toBe(length);
  expect(col.nullCount).toBe(100);
  expect(col.hasNulls).toBe(true);
  expect(col.nullable).toBe(true);
});

test('Column initialization with Array of mixed values', () => {
  const col = new Column({type: new Bool8, data: [true, null, false, null]});

  expect(col.type).toBeInstanceOf(Bool8);
  expect(col.length).toBe(4);
  expect(col.nullCount).toBe(2);
  expect(col.hasNulls).toBe(true);
  expect(col.nullable).toBe(true);
  expect(col.getValue(0)).toEqual(true);
  expect(col.getValue(1)).toEqual(null);
  expect(col.getValue(2)).toEqual(false);
  expect(col.getValue(3)).toEqual(null);
});

test('Column.gather', () => {
  const col = new Column({type: new Int32, data: new Int32Buffer([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])});

  const selection = new Column({type: new Int32, data: new Int32Buffer([2, 4, 5, 8])});

  const result = col.gather(selection, false);

  expect(result.getValue(0)).toBe(2);
  expect(result.getValue(1)).toBe(4);
  expect(result.getValue(2)).toBe(5);
  expect(result.getValue(3)).toBe(8);
});

test('Column.gather (bad argument)', () => {
  const col = new Column({type: new Int32, data: new Int32Buffer([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])});

  const selection = [2, 4, 5];

  expect(() => col.gather(<any>selection, false)).toThrow();
});

test('Column null_mask, null_count', () => {
  const length = 32;
  const col    = new Column({
    type: new Float32,
    data: new Float32Buffer(length),
    nullMask: new Uint8Buffer([254, 255, 255, 255])
  });

  expect(col.type).toBeInstanceOf(Float32);
  expect(col.length).toBe(length);
  expect(col.nullCount).toBe(1);
  expect(col.hasNulls).toBe(true);
  expect(col.nullable).toBe(true);
});

test('test child(child_index), num_children', () => {
  const utf8Col    = new Column({type: new Uint8, data: new Uint8Buffer(Buffer.from('hello'))});
  const offsetsCol = new Column({type: new Int32, data: new Int32Buffer([0, utf8Col.length])});
  const stringsCol = new Column({
    type: new Utf8String,
    length: 1,
    nullMask: new Uint8Buffer([255]),
    children: [offsetsCol, utf8Col],
  });

  expect(stringsCol.type).toBeInstanceOf(Utf8String);
  expect(stringsCol.numChildren).toBe(2);
  expect(stringsCol.getValue(0)).toBe('hello');
  expect(stringsCol.getChild(0).length).toBe(offsetsCol.length);
  expect(stringsCol.getChild(0).type).toBeInstanceOf(Int32);
  expect(stringsCol.getChild(1).length).toBe(utf8Col.length);
  expect(stringsCol.getChild(1).type).toBeInstanceOf(Uint8);
});

test('Column.dropNans', () => {
  const col    = new Column({type: new Float32(), data: new Float32Buffer([1, 3, NaN, 4, 2, 0])});
  const result = col.dropNans();

  const expected = [1, 3, 4, 2, 0];
  expect([...Series.new(result)]).toEqual(expected);
});

test('Column.dropNulls', () => {
  const mask = new Uint8Buffer(BoolVector.from([0, 1, 1, 1, 1, 0]).values);

  const col = new Column(
    {type: new Float32(), data: new Float32Buffer([1, 3, NaN, 4, 2, 0]), nullMask: mask});
  const result = col.dropNulls();

  const expected = [3, NaN, 4, 2];
  expect([...Series.new(result)]).toEqual(expected);
});

test('Column.nansToNulls', () => {
  const col = new Column({type: new Float32(), data: new Float32Buffer([1, 3, NaN, 4, 2, 0])});

  const result = col.nansToNulls();

  const expected = [1, 3, null, 4, 2, 0];
  expect([...Series.new(result)]).toEqual(expected);
});

test('Column.stringsFromBooleans', () => {
  const col      = Series.new([true, false, true, null, true])._col;
  const result   = col.stringsFromBooleans();
  const expected = ['true', 'false', 'true', null, 'true'];
  expect([...Series.new(result)]).toEqual(expected);
});

test('Column.stringsToBooleans', () => {
  const col      = Series.new(['true', 'false', 'true', null, 'true'])._col;
  const result   = col.stringsToBooleans();
  const expected = [true, false, true, null, true];
  expect([...Series.new(result)]).toEqual(expected);
});

test('Column.stringIsFloat', () => {
  const col      = Series.new(['1.2', '12', 'abc', '-2.3', '-5', null, '2e+17', '0'])._col;
  const result   = col.stringIsFloat();
  const expected = [true, true, false, true, true, null, true, true];
  expect([...Series.new(result)]).toEqual(expected);
});

test('Column.stringsFromFloats', () => {
  const col      = Series.new([1.2, 12, -2.3, -5, null, 2e+17, 0])._col;
  const result   = col.stringsFromFloats();
  const expected = ['1.2', '12.0', '-2.3', '-5.0', null, '2.0e+17', '0.0'];
  expect([...Series.new(result)]).toEqual(expected);
});

test('Column.stringsToFloats', () => {
  const col      = Series.new(['1.2', '12', '-2.3', '-5', null, '2e+17', '0'])._col;
  const result   = col.stringsToFloats(new Float64);
  const expected = [1.2, 12.0, -2.3, -5.0, null, 2.0e+17, 0.0];
  expect([...Series.new(result)]).toEqual(expected);
});

test('Column.stringIsInteger', () => {
  const col      = Series.new(['1.2', '12', 'abc', '-2.3', '-5', null, '2e+17', '0'])._col;
  const result   = col.stringIsInteger();
  const expected = [false, true, false, false, true, null, false, true];
  expect([...Series.new(result)]).toEqual(expected);
});

test('Column.stringsFromIntegers', () => {
  const col      = Series.new({type: new Int32, data: [12, -5, null, 0]})._col;
  const result   = col.stringsFromIntegers();
  const expected = ['12', '-5', null, '0'];
  expect([...Series.new(result)]).toEqual(expected);
});

test('Column.stringsToIntegers', () => {
  const col      = Series.new(['12', '-5', null, '0'])._col;
  const result   = col.stringsToIntegers(new Int32);
  const expected = [12, -5, null, 0];
  expect([...Series.new(result)]).toEqual(expected);
});

test('Column.stringIsHex', () => {
  const col      = Series.new(['123', '-456', '', 'AGE', '+17EA', '0x9EF', '123ABC', null])._col;
  const result   = col.stringIsHex();
  const expected = [true, false, false, false, false, true, true, null];
  expect([...Series.new(result)]).toEqual(expected);
});

test('Column.hexFromIntegers', () => {
  const col      = Series.new({type: new Int32, data: [1234, -1, 0, 27, 342718233, null]})._col;
  const result   = col.hexFromIntegers();
  const expected = ['04D2', 'FFFFFFFF', '00', '1B', '146D7719', null];
  expect([...Series.new(result)]).toEqual(expected);
});

test('Column.hexToIntegers', () => {
  const col      = Series.new(['04D2', 'FFFFFFFF', '00', '1B', '146D7719', null])._col;
  const result   = col.hexToIntegers(new Int32);
  const expected = [1234, -1, 0, 27, 342718233, null];
  expect([...Series.new(result)]).toEqual(expected);
});

test('Column.stringIsIpv4', () => {
  const col = Series.new(['123.255.0.7', '127.0.0.1', '', '1.2.34', '123.456.789.10', null])._col;
  const result   = col.stringIsIpv4();
  const expected = [true, true, false, false, false, null];
  expect([...Series.new(result)]).toEqual(expected);
});

test('Column.ipv4FromIntegers', () => {
  const col      = Series.new([2080309255n, 2130706433n, null])._col;
  const result   = col.ipv4FromIntegers();
  const expected = ['123.255.0.7', '127.0.0.1', null];
  expect([...Series.new(result)]).toEqual(expected);
});

test('Column.ipv4ToIntegers', () => {
  const col      = Series.new(['123.255.0.7', '127.0.0.1', null])._col;
  const result   = col.ipv4ToIntegers();
  const expected = [2080309255n, 2130706433n, null];
  expect([...Series.new(result)]).toEqual(expected);
});

describe('Column.replaceSlice', () => {
  test('prepend', () => {
    const col      = Series.new(['foo', 'bar', 'abcdef'])._col;
    const result   = col.replaceSlice('123', 0, 0);
    const expected = ['123foo', '123bar', '123abcdef'];
    expect([...Series.new(result)]).toEqual(expected);
  });
  test('append', () => {
    const col      = Series.new(['foo', 'bar', 'abcdef'])._col;
    const result   = col.replaceSlice('123', -1, -1);
    const expected = ['foo123', 'bar123', 'abcdef123'];
    expect([...Series.new(result)]).toEqual(expected);
  });
  test('insert', () => {
    const col      = Series.new(['foo', 'bar', 'abcdef'])._col;
    const result   = col.replaceSlice('123', 1, 1);
    const expected = ['f123oo', 'b123ar', 'a123bcdef'];
    expect([...Series.new(result)]).toEqual(expected);
  });
  test('replace middle', () => {
    const col      = Series.new(['foo', 'bar', 'abcdef'])._col;
    const result   = col.replaceSlice('123', 1, 2);
    const expected = ['f123o', 'b123r', 'a123cdef'];
    expect([...Series.new(result)]).toEqual(expected);
  });
  test('replace entire', () => {
    const col      = Series.new(['foo', 'bar', 'abcdef'])._col;
    const result   = col.replaceSlice('123', 0, -1);
    const expected = ['123', '123', '123'];
    expect([...Series.new(result)]).toEqual(expected);
  });
});

describe('Column.setNullMask', () => {
  const arange = (length: number) => Array.from({length}, (_, i) => i);
  const makeTestColumn            = (length: number) =>
    new Column({type: new Int32, length, data: arange(length)});

  const validateSetNullMask = (col: Column, length: number, expectedNullCount: number, newNullMask: any, ...args: [any?]) => {
    expect(col.length).toBe(length);
    expect(col.nullCount).toBe(0);

    col.setNullMask(newNullMask, ...args);

    expect(col.length).toBe(length);
    expect(col.nullCount).toBe(expectedNullCount);
    expect(col.hasNulls).toBe(expectedNullCount > 0);
    expect(col.nullable).toBe(newNullMask != null);
    if (newNullMask == null) {
      expect(col.mask.byteLength).toBe(0);
    } else {
      expect(col.mask.byteLength).toBe((((length >> 3) + 63) & ~63) || 64);
    }
  };

  test('recomputes nullCount (all null)',
       () => { validateSetNullMask(makeTestColumn(4), 4, 4, new Uint8Buffer(64).buffer); });

  test('recomputes nullCount (all valid)',
       () => { validateSetNullMask(makeTestColumn(4), 4, 0, new Uint8Buffer(8).fill(255)); });

  test('uses the new nullCount',
       () => { validateSetNullMask(makeTestColumn(4), 4, 4, new Uint8Buffer(8).buffer, 4); });

  test('clamps the new nullCount to length',
       () => { validateSetNullMask(makeTestColumn(4), 4, 4, new Uint8Buffer(8).buffer, 8); });

  test('resizes when mask is smaller than 64 bytes',
       () => { validateSetNullMask(makeTestColumn(4), 4, 4, new Uint8Buffer(4).buffer); });

  test('Passing null resets to all valid',
       () => { validateSetNullMask(makeTestColumn(4), 4, 0, null); });

  test('accepts Arrays of numbers', () => {
    validateSetNullMask(makeTestColumn(4), 4, 0, [1, 1, 1, 1]);
    validateSetNullMask(makeTestColumn(4), 4, 2, [1, 1, 0, 0]);
    validateSetNullMask(makeTestColumn(4), 4, 4, [0, 0, 0, 0]);
  });

  test('accepts Arrays of bigints', () => {
    validateSetNullMask(makeTestColumn(4), 4, 0, [1n, 1n, 1n, 1n]);
    validateSetNullMask(makeTestColumn(4), 4, 2, [1n, 1n, 0n, 0n]);
    validateSetNullMask(makeTestColumn(4), 4, 4, [0n, 0n, 0n, 0n]);
  });

  test('accepts Arrays of booleans', () => {
    validateSetNullMask(makeTestColumn(4), 4, 0, [true, true, true, true]);
    validateSetNullMask(makeTestColumn(4), 4, 2, [true, true, false, false]);
    validateSetNullMask(makeTestColumn(4), 4, 4, [false, false, false, false]);
  });

  test('accepts CUDA DeviceMemory', () => {
    validateSetNullMask(
      makeTestColumn(4), 4, 4, new Uint8Buffer(new DeviceMemory(8)).fill(0).buffer);
  });

  test('accepts CUDA MemoryView of DeviceMemory', () => {
    validateSetNullMask(makeTestColumn(4), 4, 4, new Uint8Buffer(new DeviceMemory(8)).fill(0));
  });
});
