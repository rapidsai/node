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
import {Series} from '@nvidia/cudf';
import {DeviceBuffer} from '@nvidia/rmm';
import * as arrow from 'apache-arrow';

setDefaultAllocator((byteLength: number) => new DeviceBuffer(byteLength));

const types = [
  {makeTestData: (values?: number[]) => makeTestSeries(new arrow.Int8, makeTestNumbers(values))},
  {makeTestData: (values?: number[]) => makeTestSeries(new arrow.Int16, makeTestNumbers(values))},
  {makeTestData: (values?: number[]) => makeTestSeries(new arrow.Int32, makeTestNumbers(values))},
  {makeTestData: (values?: number[]) => makeTestSeries(new arrow.Int64, makeTestBigInts(values))},
  {makeTestData: (values?: number[]) => makeTestSeries(new arrow.Uint8, makeTestNumbers(values))},
  {makeTestData: (values?: number[]) => makeTestSeries(new arrow.Uint16, makeTestNumbers(values))},
  {makeTestData: (values?: number[]) => makeTestSeries(new arrow.Uint32, makeTestNumbers(values))},
  {makeTestData: (values?: number[]) => makeTestSeries(new arrow.Uint64, makeTestBigInts(values))},
  {makeTestData: (values?: number[]) => makeTestSeries(new arrow.Float32, makeTestNumbers(values))},
  {makeTestData: (values?: number[]) => makeTestSeries(new arrow.Float64, makeTestNumbers(values))},
];

describe('Series.add', () => {
  test.each(types)('adds a Series', ({makeTestData}) => {
    const {lhs, rhs, asArray} = makeTestData();
    // lhs + lhs == [0 + 0, 1 + 1, 2 + 2]
    expect([...lhs.add(lhs)].map(Number)).toEqual(asArray([0 + 0, 1 + 1, 2 + 2]));
    // lhs + rhs == [0 + 1, 1 + 2, 2 + 3]
    expect([...lhs.add(rhs)].map(Number)).toEqual(asArray([0 + 1, 1 + 2, 2 + 3]));
  });
  test.each(types)('adds a number', ({makeTestData}) => {
    const {lhs, asArray} = makeTestData();
    expect([...lhs.add(-1)].map(Number)).toEqual(asArray([-1, 0, 1]));
    expect([...lhs.add(0)].map(Number)).toEqual(asArray([0, 1, 2]));
    expect([...lhs.add(1)].map(Number)).toEqual(asArray([1, 2, 3]));
    expect([...lhs.add(2)].map(Number)).toEqual(asArray([2, 3, 4]));
  });
  test.each(types)('adds a bigint', ({makeTestData}) => {
    const {lhs} = makeTestData();
    expect([...lhs.add(-1n)]).toEqual([-1, 0, 1].map(BigInt));
    expect([...lhs.add(0n)]).toEqual([0, 1, 2].map(BigInt));
    expect([...lhs.add(1n)]).toEqual([1, 2, 3].map(BigInt));
    expect([...lhs.add(2n)]).toEqual([2, 3, 4].map(BigInt));
  });
});

describe('Series.sub', () => {
  test.each(types)('subtracts a Series', ({makeTestData}) => {
    const {lhs, rhs, asArray} = makeTestData();
    // lhs - lhs == [0 - 0, 1 - 1, 2 - 2]
    expect([...lhs.sub(lhs)].map(Number)).toEqual(asArray([0 - 0, 1 - 1, 2 - 2]));
    // lhs - rhs == [0 - 1, 1 - 2, 2 - 3]
    expect([...lhs.sub(rhs)].map(Number)).toEqual(asArray([0 - 1, 1 - 2, 2 - 3]));
    // rhs - lhs == [1 - 0, 2 - 1, 3 - 2]
    expect([...rhs.sub(lhs)].map(Number)).toEqual(asArray([1 - 0, 2 - 1, 3 - 2]));
  });
  test.each(types)('subtracts a number', ({makeTestData}) => {
    const {lhs, asArray} = makeTestData();
    expect([...lhs.sub(-1)].map(Number)).toEqual(asArray([1, 2, 3]));
    expect([...lhs.sub(0)].map(Number)).toEqual(asArray([0, 1, 2]));
    expect([...lhs.sub(1)].map(Number)).toEqual(asArray([-1, 0, 1]));
    expect([...lhs.sub(2)].map(Number)).toEqual(asArray([-2, -1, 0]));
  });
  test.each(types)('subtracts a bigint', ({makeTestData}) => {
    const {lhs, asArray} = makeTestData();
    expect([...lhs.sub(-1n)]).toEqual(asArray([1, 2, 3].map(BigInt)));
    expect([...lhs.sub(0n)]).toEqual(asArray([0, 1, 2].map(BigInt)));
    expect([...lhs.sub(1n)]).toEqual(asArray([-1, 0, 1].map(BigInt)));
    expect([...lhs.sub(2n)]).toEqual(asArray([-2, -1, 0].map(BigInt)));
  });
});

describe('Series.mul', () => {
  test.each(types)('multiplies against a Series', ({makeTestData}) => {
    const {lhs, rhs, asArray} = makeTestData();
    // lhs * lhs == [0 * 0, 1 * 1, 2 * 2]
    expect([...lhs.mul(lhs)].map(Number)).toEqual(asArray([0 * 0, 1 * 1, 2 * 2]));
    // lhs * rhs == [0 * 1, 1 * 2, 2 * 3]
    expect([...lhs.mul(rhs)].map(Number)).toEqual(asArray([0 * 1, 1 * 2, 2 * 3]));
  });
  test.each(types)('multiplies against a number', ({makeTestData}) => {
    const {lhs, asArray} = makeTestData();
    expect([...lhs.mul(-1)].map(Number)).toEqual(asArray([-0, -1, -2]));
    expect([...lhs.mul(0)].map(Number)).toEqual(asArray([0, 0, 0]));
    expect([...lhs.mul(1)].map(Number)).toEqual(asArray([0, 1, 2]));
    expect([...lhs.mul(2)].map(Number)).toEqual(asArray([0, 2, 4]));
  });
  test.each(types)('multiplies against a bigint', ({makeTestData}) => {
    const {lhs, asArray} = makeTestData();
    expect([...lhs.mul(-1n)]).toEqual(asArray([-0, -1, -2].map(BigInt)));
    expect([...lhs.mul(0n)]).toEqual(asArray([0, 0, 0].map(BigInt)));
    expect([...lhs.mul(1n)]).toEqual(asArray([0, 1, 2].map(BigInt)));
    expect([...lhs.mul(2n)]).toEqual(asArray([0, 2, 4].map(BigInt)));
  });
});

describe('Series.div', () => {
  test.each(types)('divides by a Series', ({makeTestData}) => {
    const {lhs, rhs, asArray} = makeTestData();
    // lhs / lhs == [0/0, 1/1, 2/2]
    expect([...lhs.div(lhs)].map(Number)).toEqual(asArray([0 / 0, 1 / 1, 2 / 2]));
    // lhs / rhs == [0/1, 1/2, 2/3]
    expect([...lhs.div(rhs)].map(Number)).toEqual(asArray([0 / 1, 1 / 2, 2 / 3]));
  });
  test.each(types)('divides by a number', ({makeTestData}) => {
    const {lhs, asArray} = makeTestData();
    expect([...lhs.div(-1)].map(Number)).toEqual(asArray([-0, -1, -2]));
    expect([...lhs.div(0)].map(Number)).toEqual(asArray([NaN, Infinity, Infinity]));
    expect([...lhs.div(1)].map(Number)).toEqual(asArray([0, 1, 2]));
    expect([...lhs.div(2)].map(Number)).toEqual(asArray([0, 0.5, 1]));
  });
  test.each(types)('divides by a bigint', ({makeTestData}) => {
    const {lhs, asArray} = makeTestData();
    expect([...lhs.div(-1n)]).toEqual(asArray([-0, -1, -2].map(BigInt)));
    expect([...lhs.div(0n)]).toEqual(asArray([NaN, Infinity, Infinity].map(BigInt)));
    expect([...lhs.div(1n)]).toEqual(asArray([0, 1, 2].map(BigInt)));
    expect([...lhs.div(2n)]).toEqual(asArray([0, 0.5, 1].map(BigInt)));
  });
});

describe('Series.true_div', () => {
  test.each(types)('true_divides by a Series', ({makeTestData}) => {
    const {lhs, rhs, asArray} = makeTestData();
    // lhs / lhs == [0/0, 1/1, 2/2]
    expect([...lhs.true_div(lhs)].map(Number)).toEqual(asArray([0 / 0, 1 / 1, 2 / 2]));
    // lhs / rhs == [0/1, 1/2, 2/3]
    expect([...lhs.true_div(rhs)].map(Number)).toEqual(asArray([0 / 1, 1 / 2, 2 / 3]));
  });
  test.each(types)('true_divides by a number', ({makeTestData}) => {
    const {lhs, asArray} = makeTestData();
    expect([...lhs.true_div(-1)].map(Number)).toEqual(asArray([-0, -1, -2]));
    expect([...lhs.true_div(0)].map(Number)).toEqual(asArray([NaN, Infinity, Infinity]));
    expect([...lhs.true_div(1)].map(Number)).toEqual(asArray([0, 1, 2]));
    expect([...lhs.true_div(2)].map(Number)).toEqual(asArray([0, 0.5, 1]));
  });
  test.each(types)('true_divides by a bigint', ({makeTestData}) => {
    const {lhs, asArray} = makeTestData();
    expect([...lhs.true_div(-1)]).toEqual(asArray([-0, -1, -2].map(BigInt)));
    expect([...lhs.true_div(0)]).toEqual(asArray([NaN, Infinity, Infinity].map(BigInt)));
    expect([...lhs.true_div(1)]).toEqual(asArray([0, 1, 2].map(BigInt)));
    expect([...lhs.true_div(2)]).toEqual(asArray([0, 0.5, 1].map(BigInt)));
  });
});

describe('Series.floor_div', () => {
  test.each(types)('floor_divides by a Series', ({makeTestData}) => {
    const {lhs, rhs, asArray} = makeTestData();
    // lhs / lhs == floor([0/0, 1/1, 2/2])
    expect([...lhs.floor_div(lhs)].map(Number))
      .toEqual(asArray([0 / 0, 1 / 1, 2 / 2].map(Math.floor)));
    // lhs / rhs == floor([0/1, 1/2, 2/3])
    expect([...lhs.floor_div(rhs)].map(Number)).toEqual(asArray([0, 0, 0].map(Math.floor)));
  });
  test.each(types)('floor_divides by a number', ({makeTestData}) => {
    const {lhs, asArray} = makeTestData();
    expect([...lhs.floor_div(-1)].map(Number)).toEqual(asArray([-0, -1, -2]));
    expect([...lhs.floor_div(0)].map(Number)).toEqual(asArray([NaN, Infinity, Infinity]));
    expect([...lhs.floor_div(1)].map(Number)).toEqual(asArray([0, 1, 2]));
    expect([...lhs.floor_div(2)].map(Number)).toEqual(asArray([0, 0, 1]));
  });
  test.each(types)('floor_divides by a bigint', ({makeTestData}) => {
    const {lhs, asArray} = makeTestData();
    expect([...lhs.floor_div(-1)]).toEqual(asArray([-0, -1, -2].map(BigInt)));
    expect([...lhs.floor_div(0)]).toEqual(asArray([NaN, Infinity, Infinity].map(BigInt)));
    expect([...lhs.floor_div(1)]).toEqual(asArray([0, 1, 2].map(BigInt)));
    expect([...lhs.floor_div(2)]).toEqual(asArray([0, 0, 1].map(BigInt)));
  });
});

describe('Series.mod', () => {
  test.each(types)('modulo by a Series', ({makeTestData}) => {
    const {lhs, rhs, asArray} = makeTestData();
    // lhs % lhs == [0 % 0, 1 % 1, 2 % 2])
    expect([...lhs.mod(lhs)].map(Number)).toEqual(asArray([0 % 0, 1 % 1, 2 % 2]));
    // lhs % rhs == [0 % 1, 1 % 2, 2 % 3])
    expect([...lhs.mod(rhs)].map(Number)).toEqual(asArray([0 % 1, 1 % 2, 2 % 3]));
  });
  test.each(types)('modulo by a number', ({makeTestData}) => {
    const {lhs, asArray} = makeTestData();
    expect([...lhs.mod(-1)].map(Number)).toEqual(asArray([0, 0, 0]));
    expect([...lhs.mod(0)].map(Number)).toEqual(asArray([NaN, NaN, NaN]));
    expect([...lhs.mod(1)].map(Number)).toEqual(asArray([0, 0, 0]));
    expect([...lhs.mod(2)].map(Number)).toEqual(asArray([0, 1, 0]));
  });
  test.each(types)('modulo by a bigint', ({makeTestData}) => {
    const {lhs, asArray} = makeTestData();
    expect([...lhs.mod(-1)]).toEqual(asArray([0, 0, 0].map(BigInt)));
    expect([...lhs.mod(0)]).toEqual(asArray([NaN, NaN, NaN].map(BigInt)));
    expect([...lhs.mod(1)]).toEqual(asArray([0, 0, 0].map(BigInt)));
    expect([...lhs.mod(2)]).toEqual(asArray([0, 1, 0].map(BigInt)));
  });
});

describe('Series.pow', () => {
  test.each(types)('computes to the power of a Series', ({makeTestData}) => {
    const {lhs, rhs, asArray} = makeTestData();
    // lhs ** lhs == [0 ** 0, 1 ** 1, 2 ** 2])
    expect([...lhs.pow(lhs)].map(Number)).toEqual(asArray([0 ** 0, 1 ** 1, 2 ** 2]));
    // lhs ** rhs == [0 ** 1, 1 ** 2, 2 ** 3])
    expect([...lhs.pow(rhs)].map(Number)).toEqual(asArray([0 ** 1, 1 ** 2, 2 ** 3]));
  });
  test.each(types)('computes to the power of a number', ({makeTestData}) => {
    const {lhs, asArray} = makeTestData();
    expect([...lhs.pow(-1)].map(Number)).toEqual(asArray([Infinity, 1, 0.5]));
    expect([...lhs.pow(0)].map(Number)).toEqual(asArray([1, 1, 1]));
    expect([...lhs.pow(1)].map(Number)).toEqual(asArray([0, 1, 2]));
    expect([...lhs.pow(2)].map(Number)).toEqual(asArray([0, 1, 4]));
  });
  test.each(types)('computes to the power of a bigint', ({makeTestData}) => {
    const {lhs, asArray} = makeTestData();
    expect([...lhs.pow(-1)]).toEqual(asArray([Infinity, 1, 0.5].map(BigInt)));
    expect([...lhs.pow(0)]).toEqual(asArray([1, 1, 1].map(BigInt)));
    expect([...lhs.pow(1)]).toEqual(asArray([0, 1, 2].map(BigInt)));
    expect([...lhs.pow(2)]).toEqual(asArray([0, 1, 4].map(BigInt)));
  });
});

describe('Series.eq', () => {
  test.each(types)('compares against Series', ({makeTestData}) => {
    const {lhs, rhs} = makeTestData();
    // lhs == lhs == true
    expect([...lhs.eq(lhs)]).toEqual([true, true, true]);
    // lhs == rhs == false
    expect([...lhs.eq(rhs)]).toEqual([false, false, false]);
  });
  test.each(types)('compares against numbers', ({makeTestData}) => {
    const {lhs} = makeTestData();
    expect([...lhs.eq(0)]).toEqual([true, false, false]);
    expect([...lhs.eq(1)]).toEqual([false, true, false]);
    expect([...lhs.eq(2)]).toEqual([false, false, true]);
  });
  test.each(types)('compares against bigints', ({makeTestData}) => {
    const {lhs} = makeTestData();
    expect([...lhs.eq(0n)]).toEqual([true, false, false]);
    expect([...lhs.eq(1n)]).toEqual([false, true, false]);
    expect([...lhs.eq(2n)]).toEqual([false, false, true]);
  });
});

describe('Series.ne', () => {
  test.each(types)('compares against Series', ({makeTestData}) => {
    const {lhs, rhs} = makeTestData();
    // lhs != rhs == true
    expect([...lhs.ne(rhs)]).toEqual([true, true, true]);
    // lhs != lhs == false
    expect([...lhs.ne(lhs)]).toEqual([false, false, false]);
  });
  test.each(types)('compares against numbers', ({makeTestData}) => {
    const {lhs} = makeTestData();
    expect([...lhs.ne(0)]).toEqual([false, true, true]);
    expect([...lhs.ne(1)]).toEqual([true, false, true]);
    expect([...lhs.ne(2)]).toEqual([true, true, false]);
  });
  test.each(types)('compares against bigints', ({makeTestData}) => {
    const {lhs} = makeTestData();
    expect([...lhs.ne(0n)]).toEqual([false, true, true]);
    expect([...lhs.ne(1n)]).toEqual([true, false, true]);
    expect([...lhs.ne(2n)]).toEqual([true, true, false]);
  });
});

describe('Series.lt', () => {
  test.each(types)('compares against Series', ({makeTestData}) => {
    const {lhs, rhs} = makeTestData();
    // lhs < rhs == true
    expect([...lhs.lt(rhs)]).toEqual([true, true, true]);
    // lhs < lhs == false
    expect([...lhs.lt(lhs)]).toEqual([false, false, false]);
    // rhs < lhs == false
    expect([...rhs.lt(rhs)]).toEqual([false, false, false]);
  });
  test.each(types)('compares against numbers', ({makeTestData}) => {
    const {lhs} = makeTestData();
    expect([...lhs.lt(3)]).toEqual([true, true, true]);
    expect([...lhs.lt(2)]).toEqual([true, true, false]);
    expect([...lhs.lt(1)]).toEqual([true, false, false]);
    expect([...lhs.lt(0)]).toEqual([false, false, false]);
  });
  test.each(types)('compares against bigints', ({makeTestData}) => {
    const {lhs} = makeTestData();
    expect([...lhs.lt(3n)]).toEqual([true, true, true]);
    expect([...lhs.lt(2n)]).toEqual([true, true, false]);
    expect([...lhs.lt(1n)]).toEqual([true, false, false]);
    expect([...lhs.lt(0n)]).toEqual([false, false, false]);
  });
});

describe('Series.le', () => {
  test.each(types)('compares against Series', ({makeTestData}) => {
    const {lhs, rhs} = makeTestData();
    // lhs <= lhs == true
    expect([...lhs.le(lhs)]).toEqual([true, true, true]);
    // lhs <= rhs == true
    expect([...lhs.le(rhs)]).toEqual([true, true, true]);
    // rhs <= lhs == false
    expect([...rhs.le(lhs)]).toEqual([false, false, false]);
  });
  test.each(types)('compares against numbers', ({makeTestData}) => {
    const {lhs} = makeTestData();
    expect([...lhs.le(2)]).toEqual([true, true, true]);
    expect([...lhs.le(1)]).toEqual([true, true, false]);
    expect([...lhs.le(0)]).toEqual([true, false, false]);
  });
  test.each(types)('compares against bigints', ({makeTestData}) => {
    const {lhs} = makeTestData();
    expect([...lhs.le(2n)]).toEqual([true, true, true]);
    expect([...lhs.le(1n)]).toEqual([true, true, false]);
    expect([...lhs.le(0n)]).toEqual([true, false, false]);
  });
});

describe('Series.gt', () => {
  test.each(types)('compares against Series', ({makeTestData}) => {
    const {lhs, rhs} = makeTestData();
    // rhs > lhs == true
    expect([...rhs.gt(lhs)]).toEqual([true, true, true]);
    // lhs > rhs == false
    expect([...lhs.gt(rhs)]).toEqual([false, false, false]);
    // lhs > lhs == false
    expect([...lhs.gt(lhs)]).toEqual([false, false, false]);
  });
  test.each(types)('compares against numbers', ({makeTestData}) => {
    const {lhs} = makeTestData();
    expect([...lhs.gt(2)]).toEqual([false, false, false]);
    expect([...lhs.gt(1)]).toEqual([false, false, true]);
    expect([...lhs.gt(0)]).toEqual([false, true, true]);
    expect([...lhs.gt(-1)]).toEqual([true, true, true]);
  });
  test.each(types)('compares against bigints', ({makeTestData}) => {
    const {lhs} = makeTestData();
    expect([...lhs.gt(2n)]).toEqual([false, false, false]);
    expect([...lhs.gt(1n)]).toEqual([false, false, true]);
    expect([...lhs.gt(0n)]).toEqual([false, true, true]);
    expect([...lhs.gt(-1n)]).toEqual([true, true, true]);
  });
});

describe('Series.ge', () => {
  test.each(types)('compares against Series', ({makeTestData}) => {
    const {lhs, rhs} = makeTestData();
    // lhs >= lhs == true
    expect([...lhs.ge(lhs)]).toEqual([true, true, true]);
    // lhs >= rhs == false
    expect([...lhs.ge(rhs)]).toEqual([false, false, false]);
  });
  test.each(types)('compares against numbers', ({makeTestData}) => {
    const {lhs} = makeTestData();
    expect([...lhs.ge(3)]).toEqual([false, false, false]);
    expect([...lhs.ge(2)]).toEqual([false, false, true]);
    expect([...lhs.ge(1)]).toEqual([false, true, true]);
    expect([...lhs.ge(0)]).toEqual([true, true, true]);
  });
  test.each(types)('compares against bigints', ({makeTestData}) => {
    const {lhs} = makeTestData();
    expect([...lhs.ge(3n)]).toEqual([false, false, false]);
    expect([...lhs.ge(2n)]).toEqual([false, false, true]);
    expect([...lhs.ge(1n)]).toEqual([false, true, true]);
    expect([...lhs.ge(0n)]).toEqual([true, true, true]);
  });
});

describe('Series.bitwise_and', () => {
  test('bitwise_and with a Series', () => {
    const {lhs, rhs, asArray} = makeInt32TestData();
    // lhs ** lhs == [0 & 0, 1 & 1, 2 & 2])
    expect([...lhs.bitwise_and(lhs)]).toEqual(asArray([0 & 0, 1 & 1, 2 & 2]));
    // lhs ** rhs == [0 & 1, 1 & 2, 2 & 3])
    expect([...lhs.bitwise_and(rhs)]).toEqual(asArray([0 & 1, 1 & 2, 2 & 3]));
  });
  test('bitwise_and with a scalar', () => {
    const {lhs, asArray} = makeInt32TestData();
    expect([...lhs.bitwise_and(-1)]).toEqual(asArray([0, 0, 0]));
    expect([...lhs.bitwise_and(0)]).toEqual(asArray([NaN, NaN, NaN]));
    expect([...lhs.bitwise_and(1)]).toEqual(asArray([0, 0, 0]));
    expect([...lhs.bitwise_and(2)]).toEqual(asArray([0, 1, 0]));
  });
});

describe('Series.bitwise_or', () => {
  test('bitwise_or with a Series', () => {
    const {lhs, rhs, asArray} = makeInt32TestData();
    // lhs | lhs == [0 | 0, 1 | 1, 2 | 2])
    expect([...lhs.bitwise_or(lhs)]).toEqual(asArray([0 | 0, 1 | 1, 2 | 2]));
    // lhs | rhs == [0 | 1, 1 | 2, 2 | 3])
    expect([...lhs.bitwise_or(rhs)]).toEqual(asArray([0 | 1, 1 | 2, 2 | 3]));
  });
  test('bitwise_or with a scalar', () => {
    const {lhs, asArray} = makeInt32TestData();
    expect([...lhs.bitwise_or(-1)]).toEqual(asArray([0, 0, 0]));
    expect([...lhs.bitwise_or(0)]).toEqual(asArray([NaN, NaN, NaN]));
    expect([...lhs.bitwise_or(1)]).toEqual(asArray([0, 0, 0]));
    expect([...lhs.bitwise_or(2)]).toEqual(asArray([0, 1, 0]));
  });
});

describe('Series.bitwise_xor', () => {
  test('bitwise_xor with a Series', () => {
    const {lhs, rhs, asArray} = makeInt32TestData();
    // lhs ^ lhs == [0 ^ 0, 1 ^ 1, 2 ^ 2])
    expect([...lhs.bitwise_xor(lhs)]).toEqual(asArray([0 ^ 0, 1 ^ 1, 2 ^ 2]));
    // lhs ^ rhs == [0 ^ 1, 1 ^ 2, 2 ^ 3])
    expect([...lhs.bitwise_xor(rhs)]).toEqual(asArray([0 ^ 1, 1 ^ 2, 2 ^ 3]));
  });
  test('bitwise_or with a scalar', () => {
    const {lhs, asArray} = makeInt32TestData();
    expect([...lhs.bitwise_xor(-1)]).toEqual(asArray([0, 0, 0]));
    expect([...lhs.bitwise_xor(0)]).toEqual(asArray([NaN, NaN, NaN]));
    expect([...lhs.bitwise_xor(1)]).toEqual(asArray([0, 0, 0]));
    expect([...lhs.bitwise_xor(2)]).toEqual(asArray([0, 1, 0]));
  });
});

describe('Series.logical_and', () => {
  test('logical_and with a Series', () => {
    const {lhs, rhs, asArray} = makeInt32TestData();
    // lhs && lhs == [0 && 0, 1 && 1, 2 && 2])
    expect([...lhs.logical_and(lhs)]).toEqual(asArray([0 && 0, 1 && 1, 2 && 2]));
    // lhs && rhs == [0 && 1, 1 && 2, 2 && 3])
    expect([...lhs.logical_and(rhs)]).toEqual(asArray([0 && 1, 1 && 2, 2 && 3]));
  });
  test('logical_and with a scalar', () => {
    const {lhs, asArray} = makeInt32TestData();
    expect([...lhs.logical_and(-1)]).toEqual(asArray([0, 0, 0]));
    expect([...lhs.logical_and(0)]).toEqual(asArray([NaN, NaN, NaN]));
    expect([...lhs.logical_and(1)]).toEqual(asArray([0, 0, 0]));
    expect([...lhs.logical_and(2)]).toEqual(asArray([0, 1, 0]));
  });
});

describe('Series.logical_or', () => {
  test('logical_or with a Series', () => {
    const {lhs, rhs, asArray} = makeInt32TestData();
    // lhs || lhs == [0 || 0, 1 || 1, 2 || 2])
    expect([...lhs.logical_or(lhs)]).toEqual(asArray([0 || 0, 1 || 1, 2 || 2]));
    // lhs || rhs == [0 || 1, 1 || 2, 2 || 3])
    expect([...lhs.logical_or(rhs)]).toEqual(asArray([0 || 1, 1 || 2, 2 || 3]));
  });
  test('logical_or with a scalar', () => {
    const {lhs, asArray} = makeInt32TestData();
    expect([...lhs.logical_or(-1)]).toEqual(asArray([0, 0, 0]));
    expect([...lhs.logical_or(0)]).toEqual(asArray([NaN, NaN, NaN]));
    expect([...lhs.logical_or(1)]).toEqual(asArray([0, 0, 0]));
    expect([...lhs.logical_or(2)]).toEqual(asArray([0, 1, 0]));
  });
});

function makeTestNumbers(values = [0, 1, 2]) {
  return [
    values.map((x: number) => Number(x) + 0),
    values.map((x: number) => Number(x) + 1),
  ] as [number[], number[]];
}

function makeTestBigInts(values = [0, 1, 2]) {
  return [
    values.map((x: number) => BigInt(x) + 0n),
    values.map((x: number) => BigInt(x) + 1n),
  ] as [bigint[], bigint[]];
}

function makeTestSeries<T extends arrow.DataType>(
  type: T, [lhs, rhs]: [(number | bigint)[], (number | bigint)[]]) {
  return {
    lhs: new Series(arrow.Vector.from({type, values: lhs})),
    rhs: new Series(arrow.Vector.from({type, values: rhs})),
    asArray: (values: (number|bigint)[]) => [...arrow.Vector.from({type, values})].map(
      (x) => typeof values[0] == 'bigint' ? BigInt(x) : Number(x))
  };
}

const makeInt32TestData =
  (values?: number[]) => { return makeTestSeries(new arrow.Int32, makeTestNumbers(values)); };

const makeFloat64TestData =
  (values?: number[]) => { return makeTestSeries(new arrow.Float64, makeTestNumbers(values)); };
