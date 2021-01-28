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
import {DeviceBuffer} from '@nvidia/rmm';
import * as arrow from 'apache-arrow';

import {makeTestNumbers, makeTestSeries} from './utils';

setDefaultAllocator((byteLength: number) => new DeviceBuffer(byteLength));

const makeTestData = (values?: number[]) => makeTestSeries(new arrow.Int8, makeTestNumbers(values));

describe('Series binaryops (Int8)', () => {
  describe('Series.add', () => {
    test('adds a Series', () => {
      const {lhs, rhs} = makeTestData();
      // lhs + lhs == [0 + 0, 1 + 1, 2 + 2]
      expect([...lhs.add(lhs)]).toEqual();
      // lhs + rhs == [0 + 1, 1 + 2, 2 + 3]
      expect([...lhs.add(rhs)]).toEqual();
    });
    test('adds a number', () => {
      const {lhs} = makeTestData();
      expect([...lhs.add(-1)]).toEqual();
      expect([...lhs.add(0)]).toEqual();
      expect([...lhs.add(1)]).toEqual();
      expect([...lhs.add(2)]).toEqual();
    });
    test('adds a bigint', () => {
      const {lhs} = makeTestData();
      expect([...lhs.add(-1n)]).toEqual();
      expect([...lhs.add(0n)]).toEqual();
      expect([...lhs.add(1n)]).toEqual();
      expect([...lhs.add(2n)]).toEqual();
    });
  });

  describe('Series.sub', () => {
    test('subtracts a Series', () => {
      const {lhs, rhs} = makeTestData();
      // lhs - lhs == [0 - 0, 1 - 1, 2 - 2]
      expect([...lhs.sub(lhs)]).toEqual();
      // lhs - rhs == [0 - 1, 1 - 2, 2 - 3]
      expect([...lhs.sub(rhs)]).toEqual();
      // rhs - lhs == [1 - 0, 2 - 1, 3 - 2]
      expect([...rhs.sub(lhs)]).toEqual();
    });
    test('subtracts a number', () => {
      const {lhs} = makeTestData();
      expect([...lhs.sub(-1)]).toEqual();
      expect([...lhs.sub(0)]).toEqual();
      expect([...lhs.sub(1)]).toEqual();
      expect([...lhs.sub(2)]).toEqual();
    });
    test('subtracts a bigint', () => {
      const {lhs} = makeTestData();
      expect([...lhs.sub(-1n)]).toEqual();
      expect([...lhs.sub(0n)]).toEqual();
      expect([...lhs.sub(1n)]).toEqual();
      expect([...lhs.sub(2n)]).toEqual();
    });
  });

  describe('Series.mul', () => {
    test('multiplies against a Series', () => {
      const {lhs, rhs} = makeTestData();
      // lhs * lhs == [0 * 0, 1 * 1, 2 * 2]
      expect([...lhs.mul(lhs)]).toEqual();
      // lhs * rhs == [0 * 1, 1 * 2, 2 * 3]
      expect([...lhs.mul(rhs)]).toEqual();
    });
    test('multiplies against a number', () => {
      const {lhs} = makeTestData();
      expect([...lhs.mul(-1)]).toEqual();
      expect([...lhs.mul(0)]).toEqual();
      expect([...lhs.mul(1)]).toEqual();
      expect([...lhs.mul(2)]).toEqual();
    });
    test('multiplies against a bigint', () => {
      const {lhs} = makeTestData();
      expect([...lhs.mul(-1n)]).toEqual();
      expect([...lhs.mul(0n)]).toEqual();
      expect([...lhs.mul(1n)]).toEqual();
      expect([...lhs.mul(2n)]).toEqual();
    });
  });

  describe('Series.div', () => {
    test('divides by a Series', () => {
      const {lhs, rhs} = makeTestData();
      // lhs / lhs == [0/0, 1/1, 2/2]
      expect([...lhs.div(lhs)]).toEqual();
      // lhs / rhs == [0/1, 1/2, 2/3]
      expect([...lhs.div(rhs)]).toEqual();
    });
    test('divides by a number', () => {
      const {lhs} = makeTestData();
      expect([...lhs.div(-1)]).toEqual();
      expect([...lhs.div(0)]).toEqual();
      expect([...lhs.div(1)]).toEqual();
      expect([...lhs.div(2)]).toEqual();
    });
    test('divides by a bigint', () => {
      const {lhs} = makeTestData();
      expect([...lhs.div(-1n)]).toEqual();
      expect([...lhs.div(0n)]).toEqual();
      expect([...lhs.div(1n)]).toEqual();
      expect([...lhs.div(2n)]).toEqual();
    });
  });

  describe('Series.true_div', () => {
    test('true_divides by a Series', () => {
      const {lhs, rhs} = makeTestData();
      // lhs / lhs == [0/0, 1/1, 2/2]
      expect([...lhs.true_div(lhs)]).toEqual();
      // lhs / rhs == [0/1, 1/2, 2/3]
      expect([...lhs.true_div(rhs)]).toEqual();
    });
    test('true_divides by a number', () => {
      const {lhs} = makeTestData();
      expect([...lhs.true_div(-1)]).toEqual();
      expect([...lhs.true_div(0)]).toEqual();
      expect([...lhs.true_div(1)]).toEqual();
      expect([...lhs.true_div(2)]).toEqual();
    });
    test('true_divides by a bigint', () => {
      const {lhs} = makeTestData();
      expect([...lhs.true_div(-1n)]).toEqual();
      expect([...lhs.true_div(0n)]).toEqual();
      expect([...lhs.true_div(1n)]).toEqual();
      expect([...lhs.true_div(2n)]).toEqual();
    });
  });

  describe('Series.floor_div', () => {
    test('floor_divides by a Series', () => {
      const {lhs, rhs} = makeTestData();
      // lhs / lhs == floor([0/0, 1/1, 2/2])
      expect([...lhs.floor_div(lhs)]).toEqual();
      // lhs / rhs == floor([0/1, 1/2, 2/3])
      expect([...lhs.floor_div(rhs)]).toEqual();
    });
    test('floor_divides by a number', () => {
      const {lhs} = makeTestData();
      expect([...lhs.floor_div(-1)]).toEqual();
      expect([...lhs.floor_div(0)]).toEqual();
      expect([...lhs.floor_div(1)]).toEqual();
      expect([...lhs.floor_div(2)]).toEqual();
    });
    test('floor_divides by a bigint', () => {
      const {lhs} = makeTestData();
      expect([...lhs.floor_div(-1n)]).toEqual();
      expect([...lhs.floor_div(0n)]).toEqual();
      expect([...lhs.floor_div(1n)]).toEqual();
      expect([...lhs.floor_div(2n)]).toEqual();
    });
  });

  describe('Series.mod', () => {
    test('modulo by a Series', () => {
      const {lhs, rhs} = makeTestData();
      // lhs % lhs == [0 % 0, 1 % 1, 2 % 2])
      expect([...lhs.mod(lhs)]).toEqual();
      // lhs % rhs == [0 % 1, 1 % 2, 2 % 3])
      expect([...lhs.mod(rhs)]).toEqual();
    });
    test('modulo by a number', () => {
      const {lhs} = makeTestData();
      expect([...lhs.mod(-1)]).toEqual();
      expect([...lhs.mod(0)]).toEqual();
      expect([...lhs.mod(1)]).toEqual();
      expect([...lhs.mod(2)]).toEqual();
    });
    test('modulo by a bigint', () => {
      const {lhs} = makeTestData();
      expect([...lhs.mod(-1)]).toEqual();
      expect([...lhs.mod(0)]).toEqual();
      expect([...lhs.mod(1)]).toEqual();
      expect([...lhs.mod(2)]).toEqual();
    });
  });

  describe('Series.pow', () => {
    test('computes to the power of a Series', () => {
      const {lhs, rhs} = makeTestData();
      // lhs ** lhs == [0 ** 0, 1 ** 1, 2 ** 2])
      expect([...lhs.pow(lhs)]).toEqual();
      // lhs ** rhs == [0 ** 1, 1 ** 2, 2 ** 3])
      expect([...lhs.pow(rhs)]).toEqual();
    });
    test('computes to the power of a number', () => {
      const {lhs} = makeTestData();
      expect([...lhs.pow(-1)]).toEqual();
      expect([...lhs.pow(0)]).toEqual();
      expect([...lhs.pow(1)]).toEqual();
      expect([...lhs.pow(2)]).toEqual();
    });
    test('computes to the power of a bigint', () => {
      const {lhs} = makeTestData();
      expect([...lhs.pow(-1)]).toEqual();
      expect([...lhs.pow(0)]).toEqual();
      expect([...lhs.pow(1)]).toEqual();
      expect([...lhs.pow(2)]).toEqual();
    });
  });

  describe('Series.eq', () => {
    test('compares against Series', () => {
      const {lhs, rhs} = makeTestData();
      // lhs == lhs == true
      expect([...lhs.eq(lhs)]).toEqual();
      // lhs == rhs == false
      expect([...lhs.eq(rhs)]).toEqual();
    });
    test('compares against numbers', () => {
      const {lhs} = makeTestData();
      expect([...lhs.eq(0)]).toEqual();
      expect([...lhs.eq(1)]).toEqual();
      expect([...lhs.eq(2)]).toEqual();
    });
    test('compares against bigints', () => {
      const {lhs} = makeTestData();
      expect([...lhs.eq(0n)]).toEqual();
      expect([...lhs.eq(1n)]).toEqual();
      expect([...lhs.eq(2n)]).toEqual();
    });
  });

  describe('Series.ne', () => {
    test('compares against Series', () => {
      const {lhs, rhs} = makeTestData();
      // lhs != rhs == true
      expect([...lhs.ne(rhs)]).toEqual();
      // lhs != lhs == false
      expect([...lhs.ne(lhs)]).toEqual();
    });
    test('compares against numbers', () => {
      const {lhs} = makeTestData();
      expect([...lhs.ne(0)]).toEqual();
      expect([...lhs.ne(1)]).toEqual();
      expect([...lhs.ne(2)]).toEqual();
    });
    test('compares against bigints', () => {
      const {lhs} = makeTestData();
      expect([...lhs.ne(0n)]).toEqual();
      expect([...lhs.ne(1n)]).toEqual();
      expect([...lhs.ne(2n)]).toEqual();
    });
  });

  describe('Series.lt', () => {
    test('compares against Series', () => {
      const {lhs, rhs} = makeTestData();
      // lhs < rhs == true
      expect([...lhs.lt(rhs)]).toEqual();
      // lhs < lhs == false
      expect([...lhs.lt(lhs)]).toEqual();
      // rhs < lhs == false
      expect([...rhs.lt(rhs)]).toEqual();
    });
    test('compares against numbers', () => {
      const {lhs} = makeTestData();
      expect([...lhs.lt(3)]).toEqual();
      expect([...lhs.lt(2)]).toEqual();
      expect([...lhs.lt(1)]).toEqual();
      expect([...lhs.lt(0)]).toEqual();
    });
    test('compares against bigints', () => {
      const {lhs} = makeTestData();
      expect([...lhs.lt(3n)]).toEqual();
      expect([...lhs.lt(2n)]).toEqual();
      expect([...lhs.lt(1n)]).toEqual();
      expect([...lhs.lt(0n)]).toEqual();
    });
  });

  describe('Series.le', () => {
    test('compares against Series', () => {
      const {lhs, rhs} = makeTestData();
      // lhs <= lhs == true
      expect([...lhs.le(lhs)]).toEqual();
      // lhs <= rhs == true
      expect([...lhs.le(rhs)]).toEqual();
      // rhs <= lhs == false
      expect([...rhs.le(lhs)]).toEqual();
    });
    test('compares against numbers', () => {
      const {lhs} = makeTestData();
      expect([...lhs.le(2)]).toEqual();
      expect([...lhs.le(1)]).toEqual();
      expect([...lhs.le(0)]).toEqual();
    });
    test('compares against bigints', () => {
      const {lhs} = makeTestData();
      expect([...lhs.le(2n)]).toEqual();
      expect([...lhs.le(1n)]).toEqual();
      expect([...lhs.le(0n)]).toEqual();
    });
  });

  describe('Series.gt', () => {
    test('compares against Series', () => {
      const {lhs, rhs} = makeTestData();
      // rhs > lhs == true
      expect([...rhs.gt(lhs)]).toEqual();
      // lhs > rhs == false
      expect([...lhs.gt(rhs)]).toEqual();
      // lhs > lhs == false
      expect([...lhs.gt(lhs)]).toEqual();
    });
    test('compares against numbers', () => {
      const {lhs} = makeTestData();
      expect([...lhs.gt(2)]).toEqual();
      expect([...lhs.gt(1)]).toEqual();
      expect([...lhs.gt(0)]).toEqual();
      expect([...lhs.gt(-1)]).toEqual();
    });
    test('compares against bigints', () => {
      const {lhs} = makeTestData();
      expect([...lhs.gt(2n)]).toEqual();
      expect([...lhs.gt(1n)]).toEqual();
      expect([...lhs.gt(0n)]).toEqual();
      expect([...lhs.gt(-1n)]).toEqual();
    });
  });

  describe('Series.ge', () => {
    test('compares against Series', () => {
      const {lhs, rhs} = makeTestData();
      // lhs >= lhs == true
      expect([...lhs.ge(lhs)]).toEqual();
      // lhs >= rhs == false
      expect([...lhs.ge(rhs)]).toEqual();
    });
    test('compares against numbers', () => {
      const {lhs} = makeTestData();
      expect([...lhs.ge(3)]).toEqual();
      expect([...lhs.ge(2)]).toEqual();
      expect([...lhs.ge(1)]).toEqual();
      expect([...lhs.ge(0)]).toEqual();
    });
    test('compares against bigints', () => {
      const {lhs} = makeTestData();
      expect([...lhs.ge(3n)]).toEqual();
      expect([...lhs.ge(2n)]).toEqual();
      expect([...lhs.ge(1n)]).toEqual();
      expect([...lhs.ge(0n)]).toEqual();
    });
  });

  describe('Series.bitwise_and', () => {
    test('bitwise_and with a Series', () => {
      const {lhs, rhs} = makeTestData();
      // lhs ** lhs == [0 & 0, 1 & 1, 2 & 2])
      expect([...lhs.bitwise_and(lhs)]).toEqual();
      // lhs ** rhs == [0 & 1, 1 & 2, 2 & 3])
      expect([...lhs.bitwise_and(rhs)]).toEqual();
    });
    test('bitwise_and with a scalar', () => {
      const {lhs} = makeTestData();
      expect([...lhs.bitwise_and(-1)]).toEqual();
      expect([...lhs.bitwise_and(0)]).toEqual();
      expect([...lhs.bitwise_and(1)]).toEqual();
      expect([...lhs.bitwise_and(2)]).toEqual();
    });
  });

  describe('Series.bitwise_or', () => {
    test('bitwise_or with a Series', () => {
      const {lhs, rhs} = makeTestData();
      // lhs | lhs == [0 | 0, 1 | 1, 2 | 2])
      expect([...lhs.bitwise_or(lhs)]).toEqual();
      // lhs | rhs == [0 | 1, 1 | 2, 2 | 3])
      expect([...lhs.bitwise_or(rhs)]).toEqual();
    });
    test('bitwise_or with a scalar', () => {
      const {lhs} = makeTestData();
      expect([...lhs.bitwise_or(-1)]).toEqual();
      expect([...lhs.bitwise_or(0)]).toEqual();
      expect([...lhs.bitwise_or(1)]).toEqual();
      expect([...lhs.bitwise_or(2)]).toEqual();
    });
  });

  describe('Series.bitwise_xor', () => {
    test('bitwise_xor with a Series', () => {
      const {lhs, rhs} = makeTestData();
      // lhs ^ lhs == [0 ^ 0, 1 ^ 1, 2 ^ 2])
      expect([...lhs.bitwise_xor(lhs)]).toEqual();
      // lhs ^ rhs == [0 ^ 1, 1 ^ 2, 2 ^ 3])
      expect([...lhs.bitwise_xor(rhs)]).toEqual();
    });
    test('bitwise_or with a scalar', () => {
      const {lhs} = makeTestData();
      expect([...lhs.bitwise_xor(-1)]).toEqual();
      expect([...lhs.bitwise_xor(0)]).toEqual();
      expect([...lhs.bitwise_xor(1)]).toEqual();
      expect([...lhs.bitwise_xor(2)]).toEqual();
    });
  });

  describe('Series.logical_and', () => {
    test('logical_and with a Series', () => {
      const {lhs, rhs} = makeTestData();
      // lhs && lhs == [0 && 0, 1 && 1, 2 && 2])
      expect([...lhs.logical_and(lhs)]).toEqual();
      // lhs && rhs == [0 && 1, 1 && 2, 2 && 3])
      expect([...lhs.logical_and(rhs)]).toEqual();
    });
    test('logical_and with a scalar', () => {
      const {lhs} = makeTestData();
      expect([...lhs.logical_and(-1)]).toEqual();
      expect([...lhs.logical_and(0)]).toEqual();
      expect([...lhs.logical_and(1)]).toEqual();
      expect([...lhs.logical_and(2)]).toEqual();
    });
  });

  describe('Series.logical_or', () => {
    test('logical_or with a Series', () => {
      const {lhs, rhs} = makeTestData();
      // lhs || lhs == [0 || 0, 1 || 1, 2 || 2])
      expect([...lhs.logical_or(lhs)]).toEqual();
      // lhs || rhs == [0 || 1, 1 || 2, 2 || 3])
      expect([...lhs.logical_or(rhs)]).toEqual();
    });
    test('logical_or with a scalar', () => {
      const {lhs} = makeTestData();
      expect([...lhs.logical_or(-1)]).toEqual();
      expect([...lhs.logical_or(0)]).toEqual();
      expect([...lhs.logical_or(1)]).toEqual();
      expect([...lhs.logical_or(2)]).toEqual();
    });
  });

  describe('Series.shift_left', () => {
    test('shift_left with a Series', () => {
      const {lhs, rhs} = makeTestData();
      // lhs || lhs == [0 || 0, 1 || 1, 2 || 2])
      expect([...lhs.shift_left(lhs)]).toEqual();
      // lhs || rhs == [0 || 1, 1 || 2, 2 || 3])
      expect([...lhs.shift_left(rhs)]).toEqual();
    });
    test('shift_left with a scalar', () => {
      const {lhs} = makeTestData();
      expect([...lhs.shift_left(-1)]).toEqual();
      expect([...lhs.shift_left(0)]).toEqual();
      expect([...lhs.shift_left(1)]).toEqual();
      expect([...lhs.shift_left(2)]).toEqual();
    });
  });

  describe('Series.shift_right', () => {
    test('shift_right with a Series', () => {
      const {lhs, rhs} = makeTestData();
      // lhs || lhs == [0 || 0, 1 || 1, 2 || 2])
      expect([...lhs.shift_right(lhs)]).toEqual();
      // lhs || rhs == [0 || 1, 1 || 2, 2 || 3])
      expect([...lhs.shift_right(rhs)]).toEqual();
    });
    test('shift_right with a scalar', () => {
      const {lhs} = makeTestData();
      expect([...lhs.shift_right(-1)]).toEqual();
      expect([...lhs.shift_right(0)]).toEqual();
      expect([...lhs.shift_right(1)]).toEqual();
      expect([...lhs.shift_right(2)]).toEqual();
    });
  });

  describe('Series.shift_right_unsigned', () => {
    test('shift_right_unsigned with a Series', () => {
      const {lhs, rhs} = makeTestData();
      // lhs || lhs == [0 || 0, 1 || 1, 2 || 2])
      expect([...lhs.shift_right_unsigned(lhs)]).toEqual();
      // lhs || rhs == [0 || 1, 1 || 2, 2 || 3])
      expect([...lhs.shift_right_unsigned(rhs)]).toEqual();
    });
    test('shift_right_unsigned with a scalar', () => {
      const {lhs} = makeTestData();
      expect([...lhs.shift_right_unsigned(-1)]).toEqual();
      expect([...lhs.shift_right_unsigned(0)]).toEqual();
      expect([...lhs.shift_right_unsigned(1)]).toEqual();
      expect([...lhs.shift_right_unsigned(2)]).toEqual();
    });
  });

  describe('Series.log_base', () => {
    test('log_base with a Series', () => {
      const {lhs, rhs} = makeTestData();
      // lhs || lhs == [0 || 0, 1 || 1, 2 || 2])
      expect([...lhs.log_base(lhs)]).toEqual();
      // lhs || rhs == [0 || 1, 1 || 2, 2 || 3])
      expect([...lhs.log_base(rhs)]).toEqual();
    });
    test('log_base with a scalar', () => {
      const {lhs} = makeTestData();
      expect([...lhs.log_base(-1)]).toEqual();
      expect([...lhs.log_base(0)]).toEqual();
      expect([...lhs.log_base(1)]).toEqual();
      expect([...lhs.log_base(2)]).toEqual();
    });
  });

  describe('Series.atan2', () => {
    test('atan2 with a Series', () => {
      const {lhs, rhs} = makeTestData();
      // lhs || lhs == [0 || 0, 1 || 1, 2 || 2])
      expect([...lhs.atan2(lhs)]).toEqual();
      // lhs || rhs == [0 || 1, 1 || 2, 2 || 3])
      expect([...lhs.atan2(rhs)]).toEqual();
    });
    test('atan2 with a scalar', () => {
      const {lhs} = makeTestData();
      expect([...lhs.atan2(-1)]).toEqual();
      expect([...lhs.atan2(0)]).toEqual();
      expect([...lhs.atan2(1)]).toEqual();
      expect([...lhs.atan2(2)]).toEqual();
    });
  });

  describe('Series.null_equals', () => {
    test('null_equals with a Series', () => {
      const {lhs, rhs} = makeTestData();
      // lhs || lhs == [0 || 0, 1 || 1, 2 || 2])
      expect([...lhs.null_equals(lhs)]).toEqual();
      // lhs || rhs == [0 || 1, 1 || 2, 2 || 3])
      expect([...lhs.null_equals(rhs)]).toEqual();
    });
    test('null_equals with a scalar', () => {
      const {lhs} = makeTestData();
      expect([...lhs.null_equals(-1)]).toEqual();
      expect([...lhs.null_equals(0)]).toEqual();
      expect([...lhs.null_equals(1)]).toEqual();
      expect([...lhs.null_equals(2)]).toEqual();
    });
  });

  describe('Series.null_max', () => {
    test('null_max with a Series', () => {
      const {lhs, rhs} = makeTestData();
      // lhs || lhs == [0 || 0, 1 || 1, 2 || 2])
      expect([...lhs.null_max(lhs)]).toEqual();
      // lhs || rhs == [0 || 1, 1 || 2, 2 || 3])
      expect([...lhs.null_max(rhs)]).toEqual();
    });
    test('null_max with a scalar', () => {
      const {lhs} = makeTestData();
      expect([...lhs.null_max(-1)]).toEqual();
      expect([...lhs.null_max(0)]).toEqual();
      expect([...lhs.null_max(1)]).toEqual();
      expect([...lhs.null_max(2)]).toEqual();
    });
  });

  describe('Series.null_min', () => {
    test('null_min with a Series', () => {
      const {lhs, rhs} = makeTestData();
      // lhs || lhs == [0 || 0, 1 || 1, 2 || 2])
      expect([...lhs.null_min(lhs)]).toEqual();
      // lhs || rhs == [0 || 1, 1 || 2, 2 || 3])
      expect([...lhs.null_min(rhs)]).toEqual();
    });
    test('null_min with a scalar', () => {
      const {lhs} = makeTestData();
      expect([...lhs.null_min(-1)]).toEqual();
      expect([...lhs.null_min(0)]).toEqual();
      expect([...lhs.null_min(1)]).toEqual();
      expect([...lhs.null_min(2)]).toEqual();
    });
  });
});
