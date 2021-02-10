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

import {Float32Buffer, Int32Buffer, setDefaultAllocator, Uint8Buffer} from '@nvidia/cuda';
import {Bool8, Column, Float32, Int32, Series, Uint8, Utf8String} from '@nvidia/cudf';
import {CudaMemoryResource, DeviceBuffer} from '@nvidia/rmm';
import {BoolVector} from 'apache-arrow';

const mr = new CudaMemoryResource();

setDefaultAllocator((byteLength) => new DeviceBuffer(byteLength, mr));

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

  expect(col.type).toBeInstanceOf(Bool8);
  expect(col.length).toBe(length);
  expect(col.nullCount).toBe(100);
  expect(col.hasNulls).toBe(true);
  expect(col.nullable).toBe(true);
});

test('Column.gather', () => {
  const col = new Column({type: new Int32, data: new Int32Buffer([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])});

  const selection = new Column({type: new Int32, data: new Int32Buffer([2, 4, 5, 8])});

  const result = col.gather(selection);

  expect(result.getValue(0)).toBe(2);
  expect(result.getValue(1)).toBe(4);
  expect(result.getValue(2)).toBe(5);
  expect(result.getValue(3)).toBe(8);
});

test('Column.gather (bad argument)', () => {
  const col = new Column({type: new Int32, data: new Int32Buffer([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])});

  const selection = [2, 4, 5];

  expect(() => col.gather(<any>selection)).toThrow();
})

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
  const utf8Col    = new Column({type: new Uint8, data: new Uint8Buffer(Buffer.from("hello"))});
  const offsetsCol = new Column({type: new Int32, data: new Int32Buffer([0, utf8Col.length])});
  const stringsCol = new Column({
    type: new Utf8String,
    length: 1,
    nullMask: new Uint8Buffer([255]),
    children: [offsetsCol, utf8Col],
  });

  expect(stringsCol.type).toBeInstanceOf(Utf8String);
  expect(stringsCol.numChildren).toBe(2);
  expect(stringsCol.getValue(0)).toBe("hello");
  expect(stringsCol.getChild(0).length).toBe(offsetsCol.length);
  expect(stringsCol.getChild(0).type).toBeInstanceOf(Int32);
  expect(stringsCol.getChild(1).length).toBe(utf8Col.length);
  expect(stringsCol.getChild(1).type).toBeInstanceOf(Uint8);
});

test('Column.drop_nans', () => {
  const col    = new Column({type: new Float32(), data: new Float32Buffer([1, 3, NaN, 4, 2, 0])});
  const result = col.drop_nans();

  const expected = [1, 3, 4, 2, 0];
  expect([...Series.new(result).toArrow()]).toEqual(expected);
});

test('Column.drop_nulls', () => {
  const mask = new Uint8Buffer(BoolVector.from([0, 1, 1, 1, 1, 0]).values);

  const col = new Column(
    {type: new Float32(), data: new Float32Buffer([1, 3, NaN, 4, 2, 0]), nullMask: mask});
  const result = col.drop_nulls();

  const expected = [3, NaN, 4, 2];
  expect([...Series.new(result).toArrow()]).toEqual(expected);
});

test('Column.nans_to_nulls', () => {
  const col = new Column({type: new Float32(), data: new Float32Buffer([1, 3, NaN, 4, 2, 0])});

  const inplace = false;
  const result  = col.nans_to_nulls(false);

  const expected = [1, 3, null, 4, 2, 0];
  if (result !== undefined) {
    expect([...Series.new(result).toArrow()]).toEqual(expected);
  } else {
    expect(inplace).toEqual(true);
  }
});
