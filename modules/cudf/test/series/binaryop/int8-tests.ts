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
import {DeviceBuffer} from '@rapidsai/rmm';
import * as arrow from 'apache-arrow';

import {makeTestNumbers, makeTestSeries, toBigInt} from '../utils';

setDefaultAllocator((byteLength: number) => new DeviceBuffer(byteLength));

const makeTestData = (values?: (number|null)[]) =>
  makeTestSeries(new arrow.Int8, makeTestNumbers(values));

describe('Series binaryops (Int8)', () => {
  describe('Series.add', () => {
    test('adds a Series', () => {
      const {lhs, rhs} = makeTestData();
      // lhs + lhs == [0 + 0, 1 + 1, 2 + 2]
      expect([...lhs.add(lhs)].map(Number)).toEqual([0, 2, 4]);
      // lhs + rhs == [0 + 1, 1 + 2, 2 + 3]
      expect([...lhs.add(rhs)].map(Number)).toEqual([1, 3, 5]);
    });
    test('adds a number', () => {
      const {lhs} = makeTestData();
      expect([...lhs.add(-1)].map(Number)).toEqual([-1, 0, 1]);
      expect([...lhs.add(0)].map(Number)).toEqual([0, 1, 2]);
      expect([...lhs.add(1)].map(Number)).toEqual([1, 2, 3]);
      expect([...lhs.add(2)].map(Number)).toEqual([2, 3, 4]);
    });
    test('adds a bigint', () => {
      const {lhs} = makeTestData();
      expect([...lhs.add(-1n)].map(toBigInt)).toEqual([-1n, 0n, 1n]);
      expect([...lhs.add(0n)].map(toBigInt)).toEqual([0n, 1n, 2n]);
      expect([...lhs.add(1n)].map(toBigInt)).toEqual([1n, 2n, 3n]);
      expect([...lhs.add(2n)].map(toBigInt)).toEqual([2n, 3n, 4n]);
    });
  });

  describe('Series.sub', () => {
    test('subtracts a Series', () => {
      const {lhs, rhs} = makeTestData();
      // lhs - lhs == [0 - 0, 1 - 1, 2 - 2]
      expect([...lhs.sub(lhs)].map(Number)).toEqual([0, 0, 0]);
      // lhs - rhs == [0 - 1, 1 - 2, 2 - 3]
      expect([...lhs.sub(rhs)].map(Number)).toEqual([-1, -1, -1]);
      // rhs - lhs == [1 - 0, 2 - 1, 3 - 2]
      expect([...rhs.sub(lhs)].map(Number)).toEqual([1, 1, 1]);
    });
    test('subtracts a number', () => {
      const {lhs} = makeTestData();
      expect([...lhs.sub(-1)].map(Number)).toEqual([1, 2, 3]);
      expect([...lhs.sub(0)].map(Number)).toEqual([0, 1, 2]);
      expect([...lhs.sub(1)].map(Number)).toEqual([-1, 0, 1]);
      expect([...lhs.sub(2)].map(Number)).toEqual([-2, -1, 0]);
    });
    test('subtracts a bigint', () => {
      const {lhs} = makeTestData();
      expect([...lhs.sub(-1n)].map(toBigInt)).toEqual([1n, 2n, 3n]);
      expect([...lhs.sub(0n)].map(toBigInt)).toEqual([0n, 1n, 2n]);
      expect([...lhs.sub(1n)].map(toBigInt)).toEqual([-1n, 0n, 1n]);
      expect([...lhs.sub(2n)].map(toBigInt)).toEqual([-2n, -1n, 0n]);
    });
  });

  describe('Series.mul', () => {
    test('multiplies against a Series', () => {
      const {lhs, rhs} = makeTestData();
      // lhs * lhs == [0 * 0, 1 * 1, 2 * 2]
      expect([...lhs.mul(lhs)].map(Number)).toEqual([0, 1, 4]);
      // lhs * rhs == [0 * 1, 1 * 2, 2 * 3]
      expect([...lhs.mul(rhs)].map(Number)).toEqual([0, 2, 6]);
    });
    test('multiplies against a number', () => {
      const {lhs} = makeTestData();
      expect([...lhs.mul(-1)].map(Number)).toEqual([-0, -1, -2]);
      expect([...lhs.mul(0)].map(Number)).toEqual([0, 0, 0]);
      expect([...lhs.mul(1)].map(Number)).toEqual([0, 1, 2]);
      expect([...lhs.mul(2)].map(Number)).toEqual([0, 2, 4]);
    });
    test('multiplies against a bigint', () => {
      const {lhs} = makeTestData();
      expect([...lhs.mul(-1n)].map(toBigInt)).toEqual([0n, -1n, -2n]);
      expect([...lhs.mul(0n)].map(toBigInt)).toEqual([0n, 0n, 0n]);
      expect([...lhs.mul(1n)].map(toBigInt)).toEqual([0n, 1n, 2n]);
      expect([...lhs.mul(2n)].map(toBigInt)).toEqual([0n, 2n, 4n]);
    });
  });

  describe('Series.div', () => {
    test('divides by a Series', () => {
      const {lhs, rhs} = makeTestData();
      // lhs / lhs == [0/0, 1/1, 2/2]
      expect([...lhs.div(lhs)].map(Number)).toEqual([-1, 1, 1]);
      // lhs / rhs == [0/1, 1/2, 2/3]
      expect([...lhs.div(rhs)].map(Number)).toEqual([0, 0, 0]);
    });
    test('divides by a number', () => {
      const {lhs} = makeTestData();
      expect([...lhs.div(-1)].map(Number)).toEqual([-0, -1, -2]);
      expect([...lhs.div(0)].map(Number)).toEqual([NaN, Infinity, Infinity]);
      expect([...lhs.div(1)].map(Number)).toEqual([0, 1, 2]);
      expect([...lhs.div(2)].map(Number)).toEqual([0, 0.5, 1]);
    });
    test('divides by a bigint', () => {
      const {lhs} = makeTestData();
      expect([...lhs.div(-1n)].map(toBigInt)).toEqual([0n, -1n, -2n]);
      expect([...lhs.div(0n)].map(toBigInt)).toEqual([4294967295n, 4294967295n, 4294967295n]);
      expect([...lhs.div(1n)].map(toBigInt)).toEqual([0n, 1n, 2n]);
      expect([...lhs.div(2n)].map(toBigInt)).toEqual([0n, 0n, 1n]);
    });
  });

  describe('Series.true_div', () => {
    test('true_divides by a Series', () => {
      const {lhs, rhs} = makeTestData();
      // lhs / lhs == [0/0, 1/1, 2/2]
      expect([...lhs.true_div(lhs)].map(Number)).toEqual([0, 1, 1]);
      // lhs / rhs == [0/1, 1/2, 2/3]
      expect([...lhs.true_div(rhs)].map(Number)).toEqual([0, 0, 0]);
    });
    test('true_divides by a number', () => {
      const {lhs} = makeTestData();
      expect([...lhs.true_div(-1)].map(Number)).toEqual([-0, -1, -2]);
      expect([...lhs.true_div(0)].map(Number)).toEqual([NaN, Infinity, Infinity]);
      expect([...lhs.true_div(1)].map(Number)).toEqual([0, 1, 2]);
      expect([...lhs.true_div(2)].map(Number)).toEqual([0, 0.5, 1]);
    });
    test('true_divides by a bigint', () => {
      const {lhs} = makeTestData();
      expect([...lhs.true_div(-1n)].map(toBigInt)).toEqual([0n, -1n, -2n]);
      expect([...lhs.true_div(0n)].map(toBigInt))
        .toEqual([-9223372036854775808n, 9223372036854775807n, 9223372036854775807n]);
      expect([...lhs.true_div(1n)].map(toBigInt)).toEqual([0n, 1n, 2n]);
      expect([...lhs.true_div(2n)].map(toBigInt)).toEqual([0n, 0n, 1n]);
    });
  });

  describe('Series.floor_div', () => {
    test('floor_divides by a Series', () => {
      const {lhs, rhs} = makeTestData();
      // lhs / lhs == floor([0/0, 1/1, 2/2])
      expect([...lhs.floor_div(lhs)].map(Number)).toEqual([0, 1, 1]);
      // lhs / rhs == floor([0/1, 1/2, 2/3])
      expect([...lhs.floor_div(rhs)].map(Number)).toEqual([0, 0, 0]);
    });
    test('floor_divides by a number', () => {
      const {lhs} = makeTestData();
      expect([...lhs.floor_div(-1)].map(Number)).toEqual([-0, -1, -2]);
      expect([...lhs.floor_div(0)].map(Number)).toEqual([NaN, Infinity, Infinity]);
      expect([...lhs.floor_div(1)].map(Number)).toEqual([0, 1, 2]);
      expect([...lhs.floor_div(2)].map(Number)).toEqual([0, 0, 1]);
    });
    test('floor_divides by a bigint', () => {
      const {lhs} = makeTestData();
      expect([...lhs.floor_div(-1n)].map(toBigInt)).toEqual([0n, -1n, -2n]);
      expect([...lhs.floor_div(0n)].map(toBigInt))
        .toEqual([-9223372036854775808n, 9223372036854775807n, 9223372036854775807n]);
      expect([...lhs.floor_div(1n)].map(toBigInt)).toEqual([0n, 1n, 2n]);
      expect([...lhs.floor_div(2n)].map(toBigInt)).toEqual([0n, 0n, 1n]);
    });
  });

  describe('Series.mod', () => {
    test('modulo by a Series', () => {
      const {lhs, rhs} = makeTestData();
      // lhs % lhs == [0 % 0, 1 % 1, 2 % 2])
      expect([...lhs.mod(lhs)].map(Number)).toEqual([-1, 0, 0]);
      // lhs % rhs == [0 % 1, 1 % 2, 2 % 3])
      expect([...lhs.mod(rhs)].map(Number)).toEqual([0, 1, 2]);
    });
    test('modulo by a number', () => {
      const {lhs} = makeTestData();
      expect([...lhs.mod(-1)].map(Number)).toEqual([0, 0, 0]);
      expect([...lhs.mod(0)].map(Number)).toEqual([NaN, NaN, NaN]);
      expect([...lhs.mod(1)].map(Number)).toEqual([0, 0, 0]);
      expect([...lhs.mod(2)].map(Number)).toEqual([0, 1, 0]);
    });
    test('modulo by a bigint', () => {
      const {lhs} = makeTestData();
      expect([...lhs.mod(-1n)].map(Number)).toEqual([0, 0, 0]);
      expect([...lhs.mod(0n)].map(Number)).toEqual([4294967295, 4294967295, 4294967295]);
      expect([...lhs.mod(1n)].map(Number)).toEqual([0, 0, 0]);
      expect([...lhs.mod(2n)].map(Number)).toEqual([0, 1, 0]);
    });
  });

  describe('Series.pow', () => {
    test('computes to the power of a Series', () => {
      const {lhs, rhs} = makeTestData();
      // lhs ** lhs == [0 ** 0, 1 ** 1, 2 ** 2])
      expect([...lhs.pow(lhs)].map(Number)).toEqual([1, 1, 4]);
      // lhs ** rhs == [0 ** 1, 1 ** 2, 2 ** 3])
      expect([...lhs.pow(rhs)].map(Number)).toEqual([0, 1, 8]);
    });
    test('computes to the power of a number', () => {
      const {lhs} = makeTestData();
      expect([...lhs.pow(-1)].map(Number)).toEqual([Infinity, 1, 0.5]);
      expect([...lhs.pow(0)].map(Number)).toEqual([1, 1, 1]);
      expect([...lhs.pow(1)].map(Number)).toEqual([0, 1, 2]);
      expect([...lhs.pow(2)].map(Number)).toEqual([0, 1, 4]);
    });
    test('computes to the power of a bigint', () => {
      const {lhs} = makeTestData();
      expect([...lhs.pow(-1n)].map(Number)).toEqual([9223372036854776000, 1, 0]);
      expect([...lhs.pow(0n)].map(Number)).toEqual([1, 1, 1]);
      expect([...lhs.pow(1n)].map(Number)).toEqual([0, 1, 2]);
      expect([...lhs.pow(2n)].map(Number)).toEqual([0, 1, 4]);
    });
  });

  describe('Series.eq', () => {
    test('compares against Series', () => {
      const {lhs, rhs} = makeTestData();
      // lhs == lhs == true
      expect([...lhs.eq(lhs)].map(Number)).toEqual([1, 1, 1]);
      // lhs == rhs == false
      expect([...lhs.eq(rhs)].map(Number)).toEqual([0, 0, 0]);
    });
    test('compares against numbers', () => {
      const {lhs} = makeTestData();
      expect([...lhs.eq(0)].map(Number)).toEqual([1, 0, 0]);
      expect([...lhs.eq(1)].map(Number)).toEqual([0, 1, 0]);
      expect([...lhs.eq(2)].map(Number)).toEqual([0, 0, 1]);
    });
    test('compares against bigints', () => {
      const {lhs} = makeTestData();
      expect([...lhs.eq(0n)].map(toBigInt)).toEqual([1n, 0n, 0n]);
      expect([...lhs.eq(1n)].map(toBigInt)).toEqual([0n, 1n, 0n]);
      expect([...lhs.eq(2n)].map(toBigInt)).toEqual([0n, 0n, 1n]);
    });
  });

  describe('Series.ne', () => {
    test('compares against Series', () => {
      const {lhs, rhs} = makeTestData();
      // lhs != rhs == true
      expect([...lhs.ne(rhs)].map(Number)).toEqual([1, 1, 1]);
      // lhs != lhs == false
      expect([...lhs.ne(lhs)].map(Number)).toEqual([0, 0, 0]);
    });
    test('compares against numbers', () => {
      const {lhs} = makeTestData();
      expect([...lhs.ne(0)].map(Number)).toEqual([0, 1, 1]);
      expect([...lhs.ne(1)].map(Number)).toEqual([1, 0, 1]);
      expect([...lhs.ne(2)].map(Number)).toEqual([1, 1, 0]);
    });
    test('compares against bigints', () => {
      const {lhs} = makeTestData();
      expect([...lhs.ne(0n)].map(toBigInt)).toEqual([0n, 1n, 1n]);
      expect([...lhs.ne(1n)].map(toBigInt)).toEqual([1n, 0n, 1n]);
      expect([...lhs.ne(2n)].map(toBigInt)).toEqual([1n, 1n, 0n]);
    });
  });

  describe('Series.lt', () => {
    test('compares against Series', () => {
      const {lhs, rhs} = makeTestData();
      // lhs < rhs == true
      expect([...lhs.lt(rhs)].map(Number)).toEqual([1, 1, 1]);
      // lhs < lhs == false
      expect([...lhs.lt(lhs)].map(Number)).toEqual([0, 0, 0]);
      // rhs < lhs == false
      expect([...rhs.lt(rhs)].map(Number)).toEqual([0, 0, 0]);
    });
    test('compares against numbers', () => {
      const {lhs} = makeTestData();
      expect([...lhs.lt(3)].map(Number)).toEqual([1, 1, 1]);
      expect([...lhs.lt(2)].map(Number)).toEqual([1, 1, 0]);
      expect([...lhs.lt(1)].map(Number)).toEqual([1, 0, 0]);
      expect([...lhs.lt(0)].map(Number)).toEqual([0, 0, 0]);
    });
    test('compares against bigints', () => {
      const {lhs} = makeTestData();
      expect([...lhs.lt(3n)].map(toBigInt)).toEqual([1n, 1n, 1n]);
      expect([...lhs.lt(2n)].map(toBigInt)).toEqual([1n, 1n, 0n]);
      expect([...lhs.lt(1n)].map(toBigInt)).toEqual([1n, 0n, 0n]);
      expect([...lhs.lt(0n)].map(toBigInt)).toEqual([0n, 0n, 0n]);
    });
  });

  describe('Series.le', () => {
    test('compares against Series', () => {
      const {lhs, rhs} = makeTestData();
      // lhs <= lhs == true
      expect([...lhs.le(lhs)].map(Number)).toEqual([1, 1, 1]);
      // lhs <= rhs == true
      expect([...lhs.le(rhs)].map(Number)).toEqual([1, 1, 1]);
      // rhs <= lhs == false
      expect([...rhs.le(lhs)].map(Number)).toEqual([0, 0, 0]);
    });
    test('compares against numbers', () => {
      const {lhs} = makeTestData();
      expect([...lhs.le(2)].map(Number)).toEqual([1, 1, 1]);
      expect([...lhs.le(1)].map(Number)).toEqual([1, 1, 0]);
      expect([...lhs.le(0)].map(Number)).toEqual([1, 0, 0]);
    });
    test('compares against bigints', () => {
      const {lhs} = makeTestData();
      expect([...lhs.le(2n)].map(toBigInt)).toEqual([1n, 1n, 1n]);
      expect([...lhs.le(1n)].map(toBigInt)).toEqual([1n, 1n, 0n]);
      expect([...lhs.le(0n)].map(toBigInt)).toEqual([1n, 0n, 0n]);
    });
  });

  describe('Series.gt', () => {
    test('compares against Series', () => {
      const {lhs, rhs} = makeTestData();
      // rhs > lhs == true
      expect([...rhs.gt(lhs)].map(Number)).toEqual([1, 1, 1]);
      // lhs > rhs == false
      expect([...lhs.gt(rhs)].map(Number)).toEqual([0, 0, 0]);
      // lhs > lhs == false
      expect([...lhs.gt(lhs)].map(Number)).toEqual([0, 0, 0]);
    });
    test('compares against numbers', () => {
      const {lhs} = makeTestData();
      expect([...lhs.gt(2)].map(Number)).toEqual([0, 0, 0]);
      expect([...lhs.gt(1)].map(Number)).toEqual([0, 0, 1]);
      expect([...lhs.gt(0)].map(Number)).toEqual([0, 1, 1]);
      expect([...lhs.gt(-1)].map(Number)).toEqual([1, 1, 1]);
    });
    test('compares against bigints', () => {
      const {lhs} = makeTestData();
      expect([...lhs.gt(2n)].map(toBigInt)).toEqual([0n, 0n, 0n]);
      expect([...lhs.gt(1n)].map(toBigInt)).toEqual([0n, 0n, 1n]);
      expect([...lhs.gt(0n)].map(toBigInt)).toEqual([0n, 1n, 1n]);
      expect([...lhs.gt(-1n)].map(toBigInt)).toEqual([1n, 1n, 1n]);
    });
  });

  describe('Series.ge', () => {
    test('compares against Series', () => {
      const {lhs, rhs} = makeTestData();
      // lhs >= lhs == true
      expect([...lhs.ge(lhs)].map(Number)).toEqual([1, 1, 1]);
      // lhs >= rhs == false
      expect([...lhs.ge(rhs)].map(Number)).toEqual([0, 0, 0]);
    });
    test('compares against numbers', () => {
      const {lhs} = makeTestData();
      expect([...lhs.ge(3)].map(Number)).toEqual([0, 0, 0]);
      expect([...lhs.ge(2)].map(Number)).toEqual([0, 0, 1]);
      expect([...lhs.ge(1)].map(Number)).toEqual([0, 1, 1]);
      expect([...lhs.ge(0)].map(Number)).toEqual([1, 1, 1]);
    });
    test('compares against bigints', () => {
      const {lhs} = makeTestData();
      expect([...lhs.ge(3n)].map(toBigInt)).toEqual([0n, 0n, 0n]);
      expect([...lhs.ge(2n)].map(toBigInt)).toEqual([0n, 0n, 1n]);
      expect([...lhs.ge(1n)].map(toBigInt)).toEqual([0n, 1n, 1n]);
      expect([...lhs.ge(0n)].map(toBigInt)).toEqual([1n, 1n, 1n]);
    });
  });

  describe('Series.bitwise_and', () => {
    test('bitwise_and with a Series', () => {
      const {lhs, rhs} = makeTestData();
      // lhs ** lhs == [0 & 0, 1 & 1, 2 & 2])
      expect([...lhs.bitwise_and(lhs)].map(Number)).toEqual([0, 1, 2]);
      // lhs ** rhs == [0 & 1, 1 & 2, 2 & 3])
      expect([...lhs.bitwise_and(rhs)].map(Number)).toEqual([0, 0, 2]);
    });
    test('bitwise_and with a scalar', () => {
      const {lhs} = makeTestData();
      expect([...lhs.bitwise_and(-1)].map(Number)).toEqual([0, 1, 2]);
      expect([...lhs.bitwise_and(0)].map(Number)).toEqual([0, 0, 0]);
      expect([...lhs.bitwise_and(1)].map(Number)).toEqual([0, 1, 0]);
      expect([...lhs.bitwise_and(2)].map(Number)).toEqual([0, 0, 2]);
    });
    test('bitwise_and with a bigint', () => {
      const {lhs} = makeTestData();
      expect([...lhs.bitwise_and(-1n)].map(Number)).toEqual([0, 1, 2]);
      expect([...lhs.bitwise_and(0n)].map(Number)).toEqual([0, 0, 0]);
      expect([...lhs.bitwise_and(1n)].map(Number)).toEqual([0, 1, 0]);
      expect([...lhs.bitwise_and(2n)].map(Number)).toEqual([0, 0, 2]);
    });
  });

  describe('Series.bitwise_or', () => {
    test('bitwise_or with a Series', () => {
      const {lhs, rhs} = makeTestData();
      // lhs | lhs == [0 | 0, 1 | 1, 2 | 2])
      expect([...lhs.bitwise_or(lhs)].map(Number)).toEqual([0, 1, 2]);
      // lhs | rhs == [0 | 1, 1 | 2, 2 | 3])
      expect([...lhs.bitwise_or(rhs)].map(Number)).toEqual([1, 3, 3]);
    });
    test('bitwise_or with a scalar', () => {
      const {lhs} = makeTestData();
      expect([...lhs.bitwise_or(-1)].map(Number)).toEqual([-1, -1, -1]);
      expect([...lhs.bitwise_or(0)].map(Number)).toEqual([0, 1, 2]);
      expect([...lhs.bitwise_or(1)].map(Number)).toEqual([1, 1, 3]);
      expect([...lhs.bitwise_or(2)].map(Number)).toEqual([2, 3, 2]);
    });
    test('bitwise_or with a bigint', () => {
      const {lhs} = makeTestData();
      expect([...lhs.bitwise_or(-1n)].map(Number)).toEqual([-1, -1, -1]);
      expect([...lhs.bitwise_or(0n)].map(Number)).toEqual([0, 1, 2]);
      expect([...lhs.bitwise_or(1n)].map(Number)).toEqual([1, 1, 3]);
      expect([...lhs.bitwise_or(2n)].map(Number)).toEqual([2, 3, 2]);
    });
  });

  describe('Series.bitwise_xor', () => {
    test('bitwise_xor with a Series', () => {
      const {lhs, rhs} = makeTestData();
      // lhs ^ lhs == [0 ^ 0, 1 ^ 1, 2 ^ 2])
      expect([...lhs.bitwise_xor(lhs)].map(Number)).toEqual([0, 0, 0]);
      // lhs ^ rhs == [0 ^ 1, 1 ^ 2, 2 ^ 3])
      expect([...lhs.bitwise_xor(rhs)].map(Number)).toEqual([1, 3, 1]);
    });
    test('bitwise_xor with a scalar', () => {
      const {lhs} = makeTestData();
      expect([...lhs.bitwise_xor(-1)].map(Number)).toEqual([-1, -2, -3]);
      expect([...lhs.bitwise_xor(0)].map(Number)).toEqual([0, 1, 2]);
      expect([...lhs.bitwise_xor(1)].map(Number)).toEqual([1, 0, 3]);
      expect([...lhs.bitwise_xor(2)].map(Number)).toEqual([2, 3, 0]);
    });
    test('bitwise_xor with a bigint', () => {
      const {lhs} = makeTestData();
      expect([...lhs.bitwise_xor(-1n)].map(Number)).toEqual([-1, -2, -3]);
      expect([...lhs.bitwise_xor(0n)].map(Number)).toEqual([0, 1, 2]);
      expect([...lhs.bitwise_xor(1n)].map(Number)).toEqual([1, 0, 3]);
      expect([...lhs.bitwise_xor(2n)].map(Number)).toEqual([2, 3, 0]);
    });
  });

  describe('Series.logical_and', () => {
    test('logical_and with a Series', () => {
      const {lhs, rhs} = makeTestData();
      // lhs && lhs == [0 && 0, 1 && 1, 2 && 2])
      expect([...lhs.logical_and(lhs)].map(Number)).toEqual([0, 1, 1]);
      // lhs && rhs == [0 && 1, 1 && 2, 2 && 3])
      expect([...lhs.logical_and(rhs)].map(Number)).toEqual([0, 1, 1]);
    });
    test('logical_and with a scalar', () => {
      const {lhs} = makeTestData();
      expect([...lhs.logical_and(-1)].map(Number)).toEqual([0, 1, 1]);
      expect([...lhs.logical_and(0)].map(Number)).toEqual([0, 0, 0]);
      expect([...lhs.logical_and(1)].map(Number)).toEqual([0, 1, 1]);
      expect([...lhs.logical_and(2)].map(Number)).toEqual([0, 1, 1]);
    });
    test('logical_and with a bigint', () => {
      const {lhs} = makeTestData();
      expect([...lhs.logical_and(-1n)].map(Number)).toEqual([0, 1, 1]);
      expect([...lhs.logical_and(0n)].map(Number)).toEqual([0, 0, 0]);
      expect([...lhs.logical_and(1n)].map(Number)).toEqual([0, 1, 1]);
      expect([...lhs.logical_and(2n)].map(Number)).toEqual([0, 1, 1]);
    });
  });

  describe('Series.logical_or', () => {
    test('logical_or with a Series', () => {
      const {lhs, rhs} = makeTestData();
      // lhs || lhs == [0 || 0, 1 || 1, 2 || 2])
      expect([...lhs.logical_or(lhs)].map(Number)).toEqual([0, 1, 1]);
      // lhs || rhs == [0 || 1, 1 || 2, 2 || 3])
      expect([...lhs.logical_or(rhs)].map(Number)).toEqual([1, 1, 1]);
    });
    test('logical_or with a scalar', () => {
      const {lhs} = makeTestData();
      expect([...lhs.logical_or(-1)].map(Number)).toEqual([1, 1, 1]);
      expect([...lhs.logical_or(0)].map(Number)).toEqual([0, 1, 1]);
      expect([...lhs.logical_or(1)].map(Number)).toEqual([1, 1, 1]);
      expect([...lhs.logical_or(2)].map(Number)).toEqual([1, 1, 1]);
    });
    test('logical_or with a bigint', () => {
      const {lhs} = makeTestData();
      expect([...lhs.logical_or(-1n)].map(Number)).toEqual([1, 1, 1]);
      expect([...lhs.logical_or(0n)].map(Number)).toEqual([0, 1, 1]);
      expect([...lhs.logical_or(1n)].map(Number)).toEqual([1, 1, 1]);
      expect([...lhs.logical_or(2n)].map(Number)).toEqual([1, 1, 1]);
    });
  });

  describe('Series.shift_left', () => {
    test('shift_left with a Series', () => {
      const {lhs, rhs} = makeTestData();
      // lhs || lhs == [0 || 0, 1 || 1, 2 || 2])
      expect([...lhs.shift_left(lhs)].map(Number)).toEqual([0, 2, 8]);
      // lhs || rhs == [0 || 1, 1 || 2, 2 || 3])
      expect([...lhs.shift_left(rhs)].map(Number)).toEqual([0, 4, 16]);
    });
    test('shift_left with a scalar', () => {
      const {lhs} = makeTestData();
      expect([...lhs.shift_left(-1)].map(Number)).toEqual([0, 0, 0]);
      expect([...lhs.shift_left(0)].map(Number)).toEqual([0, 1, 2]);
      expect([...lhs.shift_left(1)].map(Number)).toEqual([0, 2, 4]);
      expect([...lhs.shift_left(2)].map(Number)).toEqual([0, 4, 8]);
    });
    test('shift_left with a bigint', () => {
      const {lhs} = makeTestData();
      expect([...lhs.shift_left(-1n)].map(Number)).toEqual([0, 0, 0]);
      expect([...lhs.shift_left(0n)].map(Number)).toEqual([0, 1, 2]);
      expect([...lhs.shift_left(1n)].map(Number)).toEqual([0, 2, 4]);
      expect([...lhs.shift_left(2n)].map(Number)).toEqual([0, 4, 8]);
    });
  });

  describe('Series.shift_right', () => {
    test('shift_right with a Series', () => {
      const {lhs, rhs} = makeTestData();
      // lhs || lhs == [0 || 0, 1 || 1, 2 || 2])
      expect([...lhs.shift_right(lhs)].map(Number)).toEqual([0, 0, 0]);
      // lhs || rhs == [0 || 1, 1 || 2, 2 || 3])
      expect([...lhs.shift_right(rhs)].map(Number)).toEqual([0, 0, 0]);
    });
    test('shift_right with a scalar', () => {
      const {lhs} = makeTestData();
      expect([...lhs.shift_right(-1)].map(Number)).toEqual([0, 0, 0]);
      expect([...lhs.shift_right(0)].map(Number)).toEqual([0, 1, 2]);
      expect([...lhs.shift_right(1)].map(Number)).toEqual([0, 0, 1]);
      expect([...lhs.shift_right(2)].map(Number)).toEqual([0, 0, 0]);
    });
  });

  describe('Series.shift_right_unsigned', () => {
    test('shift_right_unsigned with a Series', () => {
      const {lhs, rhs} = makeTestData();
      // lhs || lhs == [0 || 0, 1 || 1, 2 || 2])
      expect([...lhs.shift_right_unsigned(lhs)].map(Number)).toEqual([0, 0, 0]);
      // lhs || rhs == [0 || 1, 1 || 2, 2 || 3])
      expect([...lhs.shift_right_unsigned(rhs)].map(Number)).toEqual([0, 0, 0]);
    });
    test('shift_right_unsigned with a scalar', () => {
      const {lhs} = makeTestData();
      expect([...lhs.shift_right_unsigned(-1)].map(Number)).toEqual([0, 0, 0]);
      expect([...lhs.shift_right_unsigned(0)].map(Number)).toEqual([0, 1, 2]);
      expect([...lhs.shift_right_unsigned(1)].map(Number)).toEqual([0, 0, 1]);
      expect([...lhs.shift_right_unsigned(2)].map(Number)).toEqual([0, 0, 0]);
    });
    test('shift_right_unsigned with a bigint', () => {
      const {lhs} = makeTestData();
      expect([...lhs.shift_right_unsigned(-1n)].map(Number)).toEqual([0, 0, 0]);
      expect([...lhs.shift_right_unsigned(0n)].map(Number)).toEqual([0, 1, 2]);
      expect([...lhs.shift_right_unsigned(1n)].map(Number)).toEqual([0, 0, 1]);
      expect([...lhs.shift_right_unsigned(2n)].map(Number)).toEqual([0, 0, 0]);
    });
  });

  describe('Series.log_base', () => {
    test('log_base with a Series', () => {
      const {lhs, rhs} = makeTestData();
      // lhs || lhs == [0 || 0, 1 || 1, 2 || 2])
      expect([...lhs.log_base(lhs)].map(Number)).toEqual([0, 0, 1]);
      // lhs || rhs == [0 || 1, 1 || 2, 2 || 3])
      expect([...lhs.log_base(rhs)].map(Number)).toEqual([0, 0, 0]);
    });
    test('log_base with a scalar', () => {
      const {lhs} = makeTestData();
      expect([...lhs.log_base(-1)].map(Number)).toEqual([NaN, NaN, NaN]);
      expect([...lhs.log_base(0)].map(Number)).toEqual([NaN, -0, -0]);
      expect([...lhs.log_base(1)].map(Number)).toEqual([-Infinity, NaN, Infinity]);
      expect([...lhs.log_base(2)].map(Number)).toEqual([-Infinity, 0, 1]);
    });
    test('log_base with a bigint', () => {
      const {lhs} = makeTestData();
      expect([...lhs.log_base(-1n)].map(Number))
        .toEqual([-9223372036854776000, -9223372036854776000, -9223372036854776000]);
      expect([...lhs.log_base(0n)].map(Number)).toEqual([-9223372036854776000, 0, 0]);
      expect([...lhs.log_base(1n)].map(Number))
        .toEqual([-9223372036854776000, -9223372036854776000, 9223372036854776000]);
      expect([...lhs.log_base(2n)].map(Number)).toEqual([-9223372036854776000, 0, 1]);
    });
  });

  describe('Series.atan2', () => {
    test('atan2 with a Series', () => {
      const {lhs, rhs} = makeTestData();
      // lhs || lhs == [0 || 0, 1 || 1, 2 || 2])
      expect([...lhs.atan2(lhs)].map(Number)).toEqual([0, 0, 0]);
      // lhs || rhs == [0 || 1, 1 || 2, 2 || 3])
      expect([...lhs.atan2(rhs)].map(Number)).toEqual([0, 0, 0]);
    });
    test('atan2 with a scalar', () => {
      const {lhs} = makeTestData();
      expect([...lhs.atan2(-1)].map(Number))
        .toEqual([3.141592653589793, 2.356194490192345, 2.0344439357957027]);
      expect([...lhs.atan2(0)].map(Number)).toEqual([0, 1.5707963267948966, 1.5707963267948966]);
      expect([...lhs.atan2(1)].map(Number)).toEqual([0, 0.7853981633974483, 1.1071487177940904]);
      expect([...lhs.atan2(2)].map(Number)).toEqual([0, 0.46364760900080615, 0.7853981633974483]);
    });
    test('atan2 with a bigint', () => {
      const {lhs} = makeTestData();
      expect([...lhs.atan2(-1n)].map(Number)).toEqual([3, 2, 2]);
      expect([...lhs.atan2(0n)].map(Number)).toEqual([0, 1, 1]);
      expect([...lhs.atan2(1n)].map(Number)).toEqual([0, 0, 1]);
      expect([...lhs.atan2(2n)].map(Number)).toEqual([0, 0, 0]);
    });
  });

  describe('Series.null_equals', () => {
    test('null_equals with a Series', () => {
      const {lhs, rhs} = makeTestData();
      // lhs || lhs == [0 || 0, 1 || 1, 2 || 2])
      expect([...lhs.null_equals(lhs)].map(Number)).toEqual([1, 1, 1]);
      // lhs || rhs == [0 || 1, 1 || 2, 2 || 3])
      expect([...lhs.null_equals(rhs)].map(Number)).toEqual([0, 0, 0]);
    });
    test('null_equals with a scalar', () => {
      const {lhs} = makeTestData();
      expect([...lhs.null_equals(-1)].map(Number)).toEqual([0, 0, 0]);
      expect([...lhs.null_equals(0)].map(Number)).toEqual([1, 0, 0]);
      expect([...lhs.null_equals(1)].map(Number)).toEqual([0, 1, 0]);
      expect([...lhs.null_equals(2)].map(Number)).toEqual([0, 0, 1]);
    });
    test('null_equals with a bigint', () => {
      const {lhs} = makeTestData();
      expect([...lhs.null_equals(-1n)].map(Number)).toEqual([0, 0, 0]);
      expect([...lhs.null_equals(0n)].map(Number)).toEqual([1, 0, 0]);
      expect([...lhs.null_equals(1n)].map(Number)).toEqual([0, 1, 0]);
      expect([...lhs.null_equals(2n)].map(Number)).toEqual([0, 0, 1]);
    });
  });

  describe('Series.null_max', () => {
    test('null_max with a Series', () => {
      const {lhs, rhs} = makeTestData();
      // lhs || lhs == [0 || 0, 1 || 1, 2 || 2])
      expect([...lhs.null_max(lhs)].map(Number)).toEqual([0, 1, 2]);
      // lhs || rhs == [0 || 1, 1 || 2, 2 || 3])
      expect([...lhs.null_max(rhs)].map(Number)).toEqual([1, 2, 3]);
    });
    test('null_max with a scalar', () => {
      const {lhs} = makeTestData();
      expect([...lhs.null_max(-1)].map(Number)).toEqual([0, 1, 2]);
      expect([...lhs.null_max(0)].map(Number)).toEqual([0, 1, 2]);
      expect([...lhs.null_max(1)].map(Number)).toEqual([1, 1, 2]);
      expect([...lhs.null_max(2)].map(Number)).toEqual([2, 2, 2]);
    });
    test('null_max with a bigint', () => {
      const {lhs} = makeTestData();
      expect([...lhs.null_max(-1n)].map(Number)).toEqual([0, 1, 2]);
      expect([...lhs.null_max(0n)].map(Number)).toEqual([0, 1, 2]);
      expect([...lhs.null_max(1n)].map(Number)).toEqual([1, 1, 2]);
      expect([...lhs.null_max(2n)].map(Number)).toEqual([2, 2, 2]);
    });
  });

  describe('Series.null_min', () => {
    test('null_min with a Series', () => {
      const {lhs, rhs} = makeTestData();
      // lhs || lhs == [0 || 0, 1 || 1, 2 || 2])
      expect([...lhs.null_min(lhs)].map(Number)).toEqual([0, 1, 2]);
      // lhs || rhs == [0 || 1, 1 || 2, 2 || 3])
      expect([...lhs.null_min(rhs)].map(Number)).toEqual([0, 1, 2]);
    });
    test('null_min with a scalar', () => {
      const {lhs} = makeTestData();
      expect([...lhs.null_min(-1)].map(Number)).toEqual([-1, -1, -1]);
      expect([...lhs.null_min(0)].map(Number)).toEqual([0, 0, 0]);
      expect([...lhs.null_min(1)].map(Number)).toEqual([0, 1, 1]);
      expect([...lhs.null_min(2)].map(Number)).toEqual([0, 1, 2]);
    });
    test('null_min with a bigint', () => {
      const {lhs} = makeTestData();
      expect([...lhs.null_min(-1n)].map(Number)).toEqual([-1, -1, -1]);
      expect([...lhs.null_min(0n)].map(Number)).toEqual([0, 0, 0]);
      expect([...lhs.null_min(1n)].map(Number)).toEqual([0, 1, 1]);
      expect([...lhs.null_min(2n)].map(Number)).toEqual([0, 1, 2]);
    });
  });
});
