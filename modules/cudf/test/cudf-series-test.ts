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

import {Int32Buffer, setDefaultAllocator, Uint8Buffer} from '@nvidia/cuda';
import {Column, Int32, NullOrder, Series, TypeId, Uint8, Utf8String} from '@nvidia/cudf';
import {CudaMemoryResource, DeviceBuffer} from '@nvidia/rmm';
import {Uint8Vector, Utf8Vector} from 'apache-arrow';
import {BoolVector} from 'apache-arrow'

const mr = new CudaMemoryResource();

setDefaultAllocator((byteLength: number) => new DeviceBuffer(byteLength, mr));

test('Series initialization with properties (non-null)', () => {
  const length = 100;
  const s      = new Series({type: new Int32(), data: new Int32Buffer(length)});

  expect(s.type.id).toBe(TypeId.INT32);
  expect(s.length).toBe(length);
  expect(s.nullCount).toBe(0);
  expect(s.hasNulls).toBe(false);
  expect(s.nullable).toBe(false);
});

test('Series initialization with properties (null)', () => {
  const length = 10;
  const s      = new Series({
    type: new Int32(),
    data: new Int32Buffer(length),
    nullMask: new Uint8Buffer([250, 255]),
  });

  expect(s.type.id).toBe(TypeId.INT32);
  expect(s.length).toBe(length);
  expect(s.nullCount).toBe(2);
  expect(s.hasNulls).toBe(true);
  expect(s.nullable).toBe(true);
});

test('Series initialization with Column', () => {
  const length = 100;
  const col    = new Column({type: TypeId.INT32, data: new Int32Buffer(length)});
  const s      = new Series(col)

  expect(s.type.id).toBe(TypeId.INT32);
  expect(s.length).toBe(length);
  expect(s.nullCount).toBe(0);
  expect(s.hasNulls).toBe(false);
  expect(s.nullable).toBe(false);
});

test('test child(child_index), num_children', () => {
  const utf8Col    = new Series({type: new Uint8(), data: new Uint8Buffer(Buffer.from("hello"))});
  const offsetsCol = new Series({type: new Int32(), data: new Int32Buffer([0, utf8Col.length])});
  const stringsCol = new Series({
    type: new Utf8String(),
    length: 1,
    nullMask: new Uint8Buffer([255]),
    children: [offsetsCol, utf8Col],
  });

  expect(stringsCol.type.id).toBe(TypeId.STRING);
  expect(stringsCol.numChildren).toBe(2);
  expect(stringsCol.nullCount).toBe(0);
  expect(stringsCol.getValue(0)).toBe("hello");
  expect(stringsCol.getChild(0).length).toBe(offsetsCol.length);
  expect(stringsCol.getChild(0).type.id).toBe(offsetsCol.type.id);
  expect(stringsCol.getChild(1).length).toBe(utf8Col.length);
  expect(stringsCol.getChild(1).type.id).toBe(utf8Col.type.id);
});

test('Series.gather', () => {
  const col =
    new Series({type: new Int32(), data: new Int32Buffer([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])});

  const selection = new Series({type: new Int32(), data: new Int32Buffer([2, 4, 5, 8])});

  const result = col.gather(selection);

  expect([...result.toArrow()]).toEqual([...selection.toArrow()])
});

describe('toArrow()', () => {
  test('converts Uint8 Series to Uint8Vector', () => {
    const uint8Col = new Series({type: new Uint8(), data: new Uint8Buffer(Buffer.from("hello"))});
    const uint8Vec = uint8Col.toArrow();
    expect(uint8Vec).toBeInstanceOf(Uint8Vector);
    expect([...uint8Vec]).toEqual([...Buffer.from('hello')]);
  });
  test('converts String Series to Utf8Vector', () => {
    const utf8Col    = new Series({type: new Uint8(), data: new Uint8Buffer(Buffer.from("hello"))});
    const offsetsCol = new Series({type: new Int32(), data: new Int32Buffer([0, utf8Col.length])});
    const stringsCol = new Series({
      type: new Utf8String(),
      length: 1,
      nullMask: new Uint8Buffer([255]),
      children: [offsetsCol, utf8Col],
    });
    const utf8Vec    = stringsCol.toArrow();
    expect(utf8Vec).toBeInstanceOf(Utf8Vector);
    expect([...utf8Vec]).toEqual(['hello']);
  });
});

test('Series.orderBy (ascending, non-null)', () => {
  const col    = new Series({type: new Int32(), data: new Int32Buffer([1, 3, 5, 4, 2, 0])});
  const result = col.orderBy(true, NullOrder.BEFORE);

  const expected = [5, 0, 4, 1, 3, 2];
  expect([...result.toArrow()]).toEqual([...Buffer.from(expected)])
});

test('Series.orderBy (descending, non-null)', () => {
  const col    = new Series({type: new Int32(), data: new Int32Buffer([1, 3, 5, 4, 2, 0])});
  const result = col.orderBy(false, NullOrder.BEFORE);

  const expected = [2, 3, 1, 4, 0, 5];
  expect([...result.toArrow()]).toEqual([...Buffer.from(expected)])
});

test('Series.orderBy (ascending, null before)', () => {
  const mask = new Uint8Buffer(BoolVector.from([1, 0, 1, 1, 1, 1]).values);
  const col =
    new Series({type: new Int32(), data: new Int32Buffer([1, 3, 5, 4, 2, 0]), nullMask: mask});
  const result = col.orderBy(true, NullOrder.BEFORE);

  const expected = [1, 5, 0, 4, 3, 2];
  expect([...result.toArrow()]).toEqual([...Buffer.from(expected)])
});

test('Series.orderBy (ascending, null after)', () => {
  const mask = new Uint8Buffer(BoolVector.from([1, 0, 1, 1, 1, 1]).values);
  const col =
    new Series({type: new Int32(), data: new Int32Buffer([1, 3, 5, 4, 2, 0]), nullMask: mask});
  const result = col.orderBy(true, NullOrder.AFTER);

  const expected = [5, 0, 4, 3, 2, 1];
  expect([...result.toArrow()]).toEqual([...Buffer.from(expected)])
});

test('Series.orderBy (descendng, null before)', () => {
  const mask = new Uint8Buffer(BoolVector.from([1, 0, 1, 1, 1, 1]).values);
  const col =
    new Series({type: new Int32(), data: new Int32Buffer([1, 3, 5, 4, 2, 0]), nullMask: mask});
  const result = col.orderBy(false, NullOrder.BEFORE);

  const expected = [2, 3, 4, 0, 5, 1];

  expect([...result.toArrow()]).toEqual([...Buffer.from(expected)])
});

test('Series.orderBy (descending, null after)', () => {
  const mask = new Uint8Buffer(BoolVector.from([1, 0, 1, 1, 1, 1]).values);
  const col =
    new Series({type: new Int32(), data: new Int32Buffer([1, 3, 5, 4, 2, 0]), nullMask: mask});
  const result = col.orderBy(false, NullOrder.AFTER);

  const expected = [1, 2, 3, 4, 0, 5];
  expect([...result.toArrow()]).toEqual([...Buffer.from(expected)])
});
