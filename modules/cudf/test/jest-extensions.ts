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

import {BigIntArray, TypedArray} from '@rapidsai/cuda';
import {zip} from 'ix/iterable';

declare global {
  // eslint-disable-next-line @typescript-eslint/no-namespace
  namespace jest {
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  interface Matchers<R> {
    toEqualTypedArray<T extends TypedArray|BigIntArray>(expected: T): CustomMatcherResult;
  }
  }
}

expect.extend({
  toEqualTypedArray,
});

function toEqualTypedArray<T extends TypedArray|BigIntArray>(
  this: jest.MatcherUtils, actual: T, expected: T) {
  const failures: Error[] = [];
  if (actual instanceof Float32Array || actual instanceof Float64Array) {
    for (const [x, y] of zip<number>(<any>actual, <any>expected)) {
      try {
        (isNaN(x) && isNaN(y)) ? expect(x).toBeNaN() : expect(x).toBeCloseTo(y);
      } catch (e) { failures.push(e); }
    }
  } else {
    try {
      expect(actual).toEqual(expected);
    } catch (e) { failures.push(e); }
  }
  return {
    pass: failures.length === 0,
    message: () => failures.join('\n'),
  };
}
