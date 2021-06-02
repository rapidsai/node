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

import {makeTestBigInts, makeTestSeries} from '../utils';

setDefaultAllocator((byteLength: number) => new DeviceBuffer(byteLength));

const makeTestData = (values?: (number|null)[]) =>
  makeTestSeries(new arrow.Int64, makeTestBigInts(values));

describe('Series binaryops (Int64)', () => {
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
      expect([...lhs.add(-1n)].map(BigInt)).toEqual([-1n, 0n, 1n]);
      expect([...lhs.add(0n)].map(BigInt)).toEqual([0n, 1n, 2n]);
      expect([...lhs.add(1n)].map(BigInt)).toEqual([1n, 2n, 3n]);
      expect([...lhs.add(2n)].map(BigInt)).toEqual([2n, 3n, 4n]);
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
      expect([...lhs.sub(-1n)].map(BigInt)).toEqual([1n, 2n, 3n]);
      expect([...lhs.sub(0n)].map(BigInt)).toEqual([0n, 1n, 2n]);
      expect([...lhs.sub(1n)].map(BigInt)).toEqual([-1n, 0n, 1n]);
      expect([...lhs.sub(2n)].map(BigInt)).toEqual([-2n, -1n, 0n]);
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
      expect([...lhs.mul(-1n)].map(BigInt)).toEqual([0n, -1n, -2n]);
      expect([...lhs.mul(0n)].map(BigInt)).toEqual([0n, 0n, 0n]);
      expect([...lhs.mul(1n)].map(BigInt)).toEqual([0n, 1n, 2n]);
      expect([...lhs.mul(2n)].map(BigInt)).toEqual([0n, 2n, 4n]);
    });
  });

  describe('Series.div', () => {
    test('divides by a Series', () => {
      const {lhs, rhs} = makeTestData();
      // lhs / lhs == [0/0, 1/1, 2/2]
      expect([...lhs.div(lhs)].map(Number)).toEqual([4294967295, 1, 1]);
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
      expect([...lhs.div(-1n)].map(BigInt)).toEqual([0n, -1n, -2n]);
      expect([...lhs.div(0n)].map(BigInt)).toEqual([4294967295n, 4294967295n, 4294967295n]);
      expect([...lhs.div(1n)].map(BigInt)).toEqual([0n, 1n, 2n]);
      expect([...lhs.div(2n)].map(BigInt)).toEqual([0n, 0n, 1n]);
    });
  });

  describe('Series.trueDiv', () => {
    test('trueDivides by a Series', () => {
      const {lhs, rhs} = makeTestData();
      // lhs / lhs == [0/0, 1/1, 2/2]
      expect([...lhs.trueDiv(lhs)].map(Number)).toEqual([-9223372036854776000, 1, 1]);
      // lhs / rhs == [0/1, 1/2, 2/3]
      expect([...lhs.trueDiv(rhs)].map(Number)).toEqual([0, 0, 0]);
    });
    test('trueDivides by a number', () => {
      const {lhs} = makeTestData();
      expect([...lhs.trueDiv(-1)].map(Number)).toEqual([-0, -1, -2]);
      expect([...lhs.trueDiv(0)].map(Number)).toEqual([NaN, Infinity, Infinity]);
      expect([...lhs.trueDiv(1)].map(Number)).toEqual([0, 1, 2]);
      expect([...lhs.trueDiv(2)].map(Number)).toEqual([0, 0.5, 1]);
    });
    test('trueDivides by a bigint', () => {
      const {lhs} = makeTestData();
      expect([...lhs.trueDiv(-1n)].map(BigInt)).toEqual([0n, -1n, -2n]);
      expect([...lhs.trueDiv(0n)].map(BigInt))
        .toEqual([-9223372036854775808n, 9223372036854775807n, 9223372036854775807n]);
      expect([...lhs.trueDiv(1n)].map(BigInt)).toEqual([0n, 1n, 2n]);
      expect([...lhs.trueDiv(2n)].map(BigInt)).toEqual([0n, 0n, 1n]);
    });
  });

  describe('Series.floorDiv', () => {
    test('floorDivides by a Series', () => {
      const {lhs, rhs} = makeTestData();
      // lhs / lhs == floor([0/0, 1/1, 2/2])
      expect([...lhs.floorDiv(lhs)].map(Number)).toEqual([-9223372036854776000, 1, 1]);
      // lhs / rhs == floor([0/1, 1/2, 2/3])
      expect([...lhs.floorDiv(rhs)].map(Number)).toEqual([0, 0, 0]);
    });
    test('floorDivides by a number', () => {
      const {lhs} = makeTestData();
      expect([...lhs.floorDiv(-1)].map(Number)).toEqual([-0, -1, -2]);
      expect([...lhs.floorDiv(0)].map(Number)).toEqual([NaN, Infinity, Infinity]);
      expect([...lhs.floorDiv(1)].map(Number)).toEqual([0, 1, 2]);
      expect([...lhs.floorDiv(2)].map(Number)).toEqual([0, 0, 1]);
    });
    test('floorDivides by a bigint', () => {
      const {lhs} = makeTestData();
      expect([...lhs.floorDiv(-1n)].map(BigInt)).toEqual([0n, -1n, -2n]);
      expect([...lhs.floorDiv(0n)].map(BigInt))
        .toEqual([-9223372036854775808n, 9223372036854775807n, 9223372036854775807n]);
      expect([...lhs.floorDiv(1n)].map(BigInt)).toEqual([0n, 1n, 2n]);
      expect([...lhs.floorDiv(2n)].map(BigInt)).toEqual([0n, 0n, 1n]);
    });
  });

  describe('Series.mod', () => {
    test('modulo by a Series', () => {
      const {lhs, rhs} = makeTestData();
      // lhs % lhs == [0 % 0, 1 % 1, 2 % 2])
      expect([...lhs.mod(lhs)].map(Number)).toEqual([4294967295, 0, 0]);
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
      expect([...lhs.eq(0n)].map(BigInt)).toEqual([1n, 0n, 0n]);
      expect([...lhs.eq(1n)].map(BigInt)).toEqual([0n, 1n, 0n]);
      expect([...lhs.eq(2n)].map(BigInt)).toEqual([0n, 0n, 1n]);
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
      expect([...lhs.ne(0n)].map(BigInt)).toEqual([0n, 1n, 1n]);
      expect([...lhs.ne(1n)].map(BigInt)).toEqual([1n, 0n, 1n]);
      expect([...lhs.ne(2n)].map(BigInt)).toEqual([1n, 1n, 0n]);
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
      expect([...lhs.lt(3n)].map(BigInt)).toEqual([1n, 1n, 1n]);
      expect([...lhs.lt(2n)].map(BigInt)).toEqual([1n, 1n, 0n]);
      expect([...lhs.lt(1n)].map(BigInt)).toEqual([1n, 0n, 0n]);
      expect([...lhs.lt(0n)].map(BigInt)).toEqual([0n, 0n, 0n]);
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
      expect([...lhs.le(2n)].map(BigInt)).toEqual([1n, 1n, 1n]);
      expect([...lhs.le(1n)].map(BigInt)).toEqual([1n, 1n, 0n]);
      expect([...lhs.le(0n)].map(BigInt)).toEqual([1n, 0n, 0n]);
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
      expect([...lhs.gt(2n)].map(BigInt)).toEqual([0n, 0n, 0n]);
      expect([...lhs.gt(1n)].map(BigInt)).toEqual([0n, 0n, 1n]);
      expect([...lhs.gt(0n)].map(BigInt)).toEqual([0n, 1n, 1n]);
      expect([...lhs.gt(-1n)].map(BigInt)).toEqual([1n, 1n, 1n]);
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
      expect([...lhs.ge(3n)].map(BigInt)).toEqual([0n, 0n, 0n]);
      expect([...lhs.ge(2n)].map(BigInt)).toEqual([0n, 0n, 1n]);
      expect([...lhs.ge(1n)].map(BigInt)).toEqual([0n, 1n, 1n]);
      expect([...lhs.ge(0n)].map(BigInt)).toEqual([1n, 1n, 1n]);
    });
  });

  describe('Series.bitwiseAnd', () => {
    test('bitwiseAnd with a Series', () => {
      const {lhs, rhs} = makeTestData();
      // lhs ** lhs == [0 & 0, 1 & 1, 2 & 2])
      expect([...lhs.bitwiseAnd(lhs)].map(Number)).toEqual([0, 1, 2]);
      // lhs ** rhs == [0 & 1, 1 & 2, 2 & 3])
      expect([...lhs.bitwiseAnd(rhs)].map(Number)).toEqual([0, 0, 2]);
    });
    test('bitwiseAnd with a scalar', () => {
      const {lhs} = makeTestData();
      expect([...lhs.bitwiseAnd(-1)].map(Number)).toEqual([0, 1, 2]);
      expect([...lhs.bitwiseAnd(0)].map(Number)).toEqual([0, 0, 0]);
      expect([...lhs.bitwiseAnd(1)].map(Number)).toEqual([0, 1, 0]);
      expect([...lhs.bitwiseAnd(2)].map(Number)).toEqual([0, 0, 2]);
    });
    test('bitwiseAnd with a bigint', () => {
      const {lhs} = makeTestData();
      expect([...lhs.bitwiseAnd(-1n)].map(Number)).toEqual([0, 1, 2]);
      expect([...lhs.bitwiseAnd(0n)].map(Number)).toEqual([0, 0, 0]);
      expect([...lhs.bitwiseAnd(1n)].map(Number)).toEqual([0, 1, 0]);
      expect([...lhs.bitwiseAnd(2n)].map(Number)).toEqual([0, 0, 2]);
    });
  });

  describe('Series.bitwiseOr', () => {
    test('bitwiseOr with a Series', () => {
      const {lhs, rhs} = makeTestData();
      // lhs | lhs == [0 | 0, 1 | 1, 2 | 2])
      expect([...lhs.bitwiseOr(lhs)].map(Number)).toEqual([0, 1, 2]);
      // lhs | rhs == [0 | 1, 1 | 2, 2 | 3])
      expect([...lhs.bitwiseOr(rhs)].map(Number)).toEqual([1, 3, 3]);
    });
    test('bitwiseOr with a scalar', () => {
      const {lhs} = makeTestData();
      expect([...lhs.bitwiseOr(-1)].map(Number)).toEqual([-1, -1, -1]);
      expect([...lhs.bitwiseOr(0)].map(Number)).toEqual([0, 1, 2]);
      expect([...lhs.bitwiseOr(1)].map(Number)).toEqual([1, 1, 3]);
      expect([...lhs.bitwiseOr(2)].map(Number)).toEqual([2, 3, 2]);
    });
    test('bitwiseOr with a bigint', () => {
      const {lhs} = makeTestData();
      expect([...lhs.bitwiseOr(-1n)].map(Number)).toEqual([-1, -1, -1]);
      expect([...lhs.bitwiseOr(0n)].map(Number)).toEqual([0, 1, 2]);
      expect([...lhs.bitwiseOr(1n)].map(Number)).toEqual([1, 1, 3]);
      expect([...lhs.bitwiseOr(2n)].map(Number)).toEqual([2, 3, 2]);
    });
  });

  describe('Series.bitwiseXor', () => {
    test('bitwiseXor with a Series', () => {
      const {lhs, rhs} = makeTestData();
      // lhs ^ lhs == [0 ^ 0, 1 ^ 1, 2 ^ 2])
      expect([...lhs.bitwiseXor(lhs)].map(Number)).toEqual([0, 0, 0]);
      // lhs ^ rhs == [0 ^ 1, 1 ^ 2, 2 ^ 3])
      expect([...lhs.bitwiseXor(rhs)].map(Number)).toEqual([1, 3, 1]);
    });
    test('bitwiseXor with a scalar', () => {
      const {lhs} = makeTestData();
      expect([...lhs.bitwiseXor(-1)].map(Number)).toEqual([-1, -2, -3]);
      expect([...lhs.bitwiseXor(0)].map(Number)).toEqual([0, 1, 2]);
      expect([...lhs.bitwiseXor(1)].map(Number)).toEqual([1, 0, 3]);
      expect([...lhs.bitwiseXor(2)].map(Number)).toEqual([2, 3, 0]);
    });
    test('bitwiseXor with a bigint', () => {
      const {lhs} = makeTestData();
      expect([...lhs.bitwiseXor(-1n)].map(Number)).toEqual([-1, -2, -3]);
      expect([...lhs.bitwiseXor(0n)].map(Number)).toEqual([0, 1, 2]);
      expect([...lhs.bitwiseXor(1n)].map(Number)).toEqual([1, 0, 3]);
      expect([...lhs.bitwiseXor(2n)].map(Number)).toEqual([2, 3, 0]);
    });
  });

  describe('Series.logicalAnd', () => {
    test('logicalAnd with a Series', () => {
      const {lhs, rhs} = makeTestData();
      // lhs && lhs == [0 && 0, 1 && 1, 2 && 2])
      expect([...lhs.logicalAnd(lhs)].map(Number)).toEqual([0, 1, 1]);
      // lhs && rhs == [0 && 1, 1 && 2, 2 && 3])
      expect([...lhs.logicalAnd(rhs)].map(Number)).toEqual([0, 1, 1]);
    });
    test('logicalAnd with a scalar', () => {
      const {lhs} = makeTestData();
      expect([...lhs.logicalAnd(-1)].map(Number)).toEqual([0, 1, 1]);
      expect([...lhs.logicalAnd(0)].map(Number)).toEqual([0, 0, 0]);
      expect([...lhs.logicalAnd(1)].map(Number)).toEqual([0, 1, 1]);
      expect([...lhs.logicalAnd(2)].map(Number)).toEqual([0, 1, 1]);
    });
    test('logicalAnd with a bigint', () => {
      const {lhs} = makeTestData();
      expect([...lhs.logicalAnd(-1n)].map(Number)).toEqual([0, 1, 1]);
      expect([...lhs.logicalAnd(0n)].map(Number)).toEqual([0, 0, 0]);
      expect([...lhs.logicalAnd(1n)].map(Number)).toEqual([0, 1, 1]);
      expect([...lhs.logicalAnd(2n)].map(Number)).toEqual([0, 1, 1]);
    });
  });

  describe('Series.logicalOr', () => {
    test('logicalOr with a Series', () => {
      const {lhs, rhs} = makeTestData();
      // lhs || lhs == [0 || 0, 1 || 1, 2 || 2])
      expect([...lhs.logicalOr(lhs)].map(Number)).toEqual([0, 1, 1]);
      // lhs || rhs == [0 || 1, 1 || 2, 2 || 3])
      expect([...lhs.logicalOr(rhs)].map(Number)).toEqual([1, 1, 1]);
    });
    test('logicalOr with a scalar', () => {
      const {lhs} = makeTestData();
      expect([...lhs.logicalOr(-1)].map(Number)).toEqual([1, 1, 1]);
      expect([...lhs.logicalOr(0)].map(Number)).toEqual([0, 1, 1]);
      expect([...lhs.logicalOr(1)].map(Number)).toEqual([1, 1, 1]);
      expect([...lhs.logicalOr(2)].map(Number)).toEqual([1, 1, 1]);
    });
    test('logicalOr with a bigint', () => {
      const {lhs} = makeTestData();
      expect([...lhs.logicalOr(-1n)].map(Number)).toEqual([1, 1, 1]);
      expect([...lhs.logicalOr(0n)].map(Number)).toEqual([0, 1, 1]);
      expect([...lhs.logicalOr(1n)].map(Number)).toEqual([1, 1, 1]);
      expect([...lhs.logicalOr(2n)].map(Number)).toEqual([1, 1, 1]);
    });
  });

  describe('Series.shiftLeft', () => {
    test('shiftLeft with a Series', () => {
      const {lhs, rhs} = makeTestData();
      // lhs || lhs == [0 || 0, 1 || 1, 2 || 2])
      expect([...lhs.shiftLeft(lhs)].map(Number)).toEqual([0, 2, 8]);
      // lhs || rhs == [0 || 1, 1 || 2, 2 || 3])
      expect([...lhs.shiftLeft(rhs)].map(Number)).toEqual([0, 4, 16]);
    });
    test('shiftLeft with a scalar', () => {
      const {lhs} = makeTestData();
      expect([...lhs.shiftLeft(-1)].map(Number)).toEqual([0, 0, 0]);
      expect([...lhs.shiftLeft(0)].map(Number)).toEqual([0, 1, 2]);
      expect([...lhs.shiftLeft(1)].map(Number)).toEqual([0, 2, 4]);
      expect([...lhs.shiftLeft(2)].map(Number)).toEqual([0, 4, 8]);
    });
    test('shiftLeft with a bigint', () => {
      const {lhs} = makeTestData();
      expect([...lhs.shiftLeft(-1n)].map(Number)).toEqual([0, 0, 0]);
      expect([...lhs.shiftLeft(0n)].map(Number)).toEqual([0, 1, 2]);
      expect([...lhs.shiftLeft(1n)].map(Number)).toEqual([0, 2, 4]);
      expect([...lhs.shiftLeft(2n)].map(Number)).toEqual([0, 4, 8]);
    });
  });

  describe('Series.shiftRight', () => {
    test('shiftRight with a Series', () => {
      const {lhs, rhs} = makeTestData();
      // lhs || lhs == [0 || 0, 1 || 1, 2 || 2])
      expect([...lhs.shiftRight(lhs)].map(Number)).toEqual([0, 0, 0]);
      // lhs || rhs == [0 || 1, 1 || 2, 2 || 3])
      expect([...lhs.shiftRight(rhs)].map(Number)).toEqual([0, 0, 0]);
    });
    test('shiftRight with a scalar', () => {
      const {lhs} = makeTestData();
      expect([...lhs.shiftRight(-1)].map(Number)).toEqual([0, 0, 0]);
      expect([...lhs.shiftRight(0)].map(Number)).toEqual([0, 1, 2]);
      expect([...lhs.shiftRight(1)].map(Number)).toEqual([0, 0, 1]);
      expect([...lhs.shiftRight(2)].map(Number)).toEqual([0, 0, 0]);
    });
    test('shiftRight with a bigint', () => {
      const {lhs} = makeTestData();
      expect([...lhs.shiftRight(-1n)].map(Number)).toEqual([0, 0, 0]);
      expect([...lhs.shiftRight(0n)].map(Number)).toEqual([0, 1, 2]);
      expect([...lhs.shiftRight(1n)].map(Number)).toEqual([0, 0, 1]);
      expect([...lhs.shiftRight(2n)].map(Number)).toEqual([0, 0, 0]);
    });
  });

  describe('Series.shiftRightUnsigned', () => {
    test('shiftRightUnsigned with a Series', () => {
      const {lhs, rhs} = makeTestData();
      // lhs || lhs == [0 || 0, 1 || 1, 2 || 2])
      expect([...lhs.shiftRightUnsigned(lhs)].map(Number)).toEqual([0, 0, 0]);
      // lhs || rhs == [0 || 1, 1 || 2, 2 || 3])
      expect([...lhs.shiftRightUnsigned(rhs)].map(Number)).toEqual([0, 0, 0]);
    });
    test('shiftRightUnsigned with a scalar', () => {
      const {lhs} = makeTestData();
      expect([...lhs.shiftRightUnsigned(-1)].map(Number)).toEqual([0, 0, 0]);
      expect([...lhs.shiftRightUnsigned(0)].map(Number)).toEqual([0, 1, 2]);
      expect([...lhs.shiftRightUnsigned(1)].map(Number)).toEqual([0, 0, 1]);
      expect([...lhs.shiftRightUnsigned(2)].map(Number)).toEqual([0, 0, 0]);
    });
    test('shiftRightUnsigned with a bigint', () => {
      const {lhs} = makeTestData();
      expect([...lhs.shiftRightUnsigned(-1n)].map(Number)).toEqual([0, 0, 0]);
      expect([...lhs.shiftRightUnsigned(0n)].map(Number)).toEqual([0, 1, 2]);
      expect([...lhs.shiftRightUnsigned(1n)].map(Number)).toEqual([0, 0, 1]);
      expect([...lhs.shiftRightUnsigned(2n)].map(Number)).toEqual([0, 0, 0]);
    });
  });

  describe('Series.logBase', () => {
    test('logBase with a Series', () => {
      const {lhs, rhs} = makeTestData();
      // lhs || lhs == [0 || 0, 1 || 1, 2 || 2])
      expect([...lhs.logBase(lhs)].map(Number))
        .toEqual([-9223372036854776000, -9223372036854776000, 1]);
      // lhs || rhs == [0 || 1, 1 || 2, 2 || 3])
      expect([...lhs.logBase(rhs)].map(Number)).toEqual([-9223372036854776000, 0, 0]);
    });
    test('logBase with a scalar', () => {
      const {lhs} = makeTestData();
      expect([...lhs.logBase(-1)].map(Number)).toEqual([NaN, NaN, NaN]);
      expect([...lhs.logBase(0)].map(Number)).toEqual([NaN, -0, -0]);
      expect([...lhs.logBase(1)].map(Number)).toEqual([-Infinity, NaN, Infinity]);
      expect([...lhs.logBase(2)].map(Number)).toEqual([-Infinity, 0, 1]);
    });
    test('logBase with a bigint', () => {
      const {lhs} = makeTestData();
      expect([...lhs.logBase(-1n)].map(Number))
        .toEqual([-9223372036854776000, -9223372036854776000, -9223372036854776000]);
      expect([...lhs.logBase(0n)].map(Number)).toEqual([-9223372036854776000, 0, 0]);
      expect([...lhs.logBase(1n)].map(Number))
        .toEqual([-9223372036854776000, -9223372036854776000, 9223372036854776000]);
      expect([...lhs.logBase(2n)].map(Number)).toEqual([-9223372036854776000, 0, 1]);
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

  describe('Series.nullEquals', () => {
    test('nullEquals with a Series', () => {
      const {lhs, rhs} = makeTestData();
      // lhs || lhs == [0 || 0, 1 || 1, 2 || 2])
      expect([...lhs.nullEquals(lhs)].map(Number)).toEqual([1, 1, 1]);
      // lhs || rhs == [0 || 1, 1 || 2, 2 || 3])
      expect([...lhs.nullEquals(rhs)].map(Number)).toEqual([0, 0, 0]);
    });
    test('nullEquals with a scalar', () => {
      const {lhs} = makeTestData();
      expect([...lhs.nullEquals(-1)].map(Number)).toEqual([0, 0, 0]);
      expect([...lhs.nullEquals(0)].map(Number)).toEqual([1, 0, 0]);
      expect([...lhs.nullEquals(1)].map(Number)).toEqual([0, 1, 0]);
      expect([...lhs.nullEquals(2)].map(Number)).toEqual([0, 0, 1]);
    });
    test('nullEquals with a bigint', () => {
      const {lhs} = makeTestData();
      expect([...lhs.nullEquals(-1n)].map(Number)).toEqual([0, 0, 0]);
      expect([...lhs.nullEquals(0n)].map(Number)).toEqual([1, 0, 0]);
      expect([...lhs.nullEquals(1n)].map(Number)).toEqual([0, 1, 0]);
      expect([...lhs.nullEquals(2n)].map(Number)).toEqual([0, 0, 1]);
    });
  });

  describe('Series.nullMax', () => {
    test('nullMax with a Series', () => {
      const {lhs, rhs} = makeTestData();
      // lhs || lhs == [0 || 0, 1 || 1, 2 || 2])
      expect([...lhs.nullMax(lhs)].map(Number)).toEqual([0, 1, 2]);
      // lhs || rhs == [0 || 1, 1 || 2, 2 || 3])
      expect([...lhs.nullMax(rhs)].map(Number)).toEqual([1, 2, 3]);
    });
    test('nullMax with a scalar', () => {
      const {lhs} = makeTestData();
      expect([...lhs.nullMax(-1)].map(Number)).toEqual([0, 1, 2]);
      expect([...lhs.nullMax(0)].map(Number)).toEqual([0, 1, 2]);
      expect([...lhs.nullMax(1)].map(Number)).toEqual([1, 1, 2]);
      expect([...lhs.nullMax(2)].map(Number)).toEqual([2, 2, 2]);
    });
    test('nullMax with a bigint', () => {
      const {lhs} = makeTestData();
      expect([...lhs.nullMax(-1n)].map(Number)).toEqual([0, 1, 2]);
      expect([...lhs.nullMax(0n)].map(Number)).toEqual([0, 1, 2]);
      expect([...lhs.nullMax(1n)].map(Number)).toEqual([1, 1, 2]);
      expect([...lhs.nullMax(2n)].map(Number)).toEqual([2, 2, 2]);
    });
  });

  describe('Series.nullMin', () => {
    test('nullMin with a Series', () => {
      const {lhs, rhs} = makeTestData();
      // lhs || lhs == [0 || 0, 1 || 1, 2 || 2])
      expect([...lhs.nullMin(lhs)].map(Number)).toEqual([0, 1, 2]);
      // lhs || rhs == [0 || 1, 1 || 2, 2 || 3])
      expect([...lhs.nullMin(rhs)].map(Number)).toEqual([0, 1, 2]);
    });
    test('nullMin with a scalar', () => {
      const {lhs} = makeTestData();
      expect([...lhs.nullMin(-1)].map(Number)).toEqual([-1, -1, -1]);
      expect([...lhs.nullMin(0)].map(Number)).toEqual([0, 0, 0]);
      expect([...lhs.nullMin(1)].map(Number)).toEqual([0, 1, 1]);
      expect([...lhs.nullMin(2)].map(Number)).toEqual([0, 1, 2]);
    });
    test('nullMin with a bigint', () => {
      const {lhs} = makeTestData();
      expect([...lhs.nullMin(-1n)].map(Number)).toEqual([-1, -1, -1]);
      expect([...lhs.nullMin(0n)].map(Number)).toEqual([0, 0, 0]);
      expect([...lhs.nullMin(1n)].map(Number)).toEqual([0, 1, 1]);
      expect([...lhs.nullMin(2n)].map(Number)).toEqual([0, 1, 2]);
    });
  });
});
