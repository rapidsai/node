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

import {
  DataFrame,
  Float32,
  Int32,
  Series,
} from '@rapidsai/cudf';

describe('dataframe.sum', () => {
  test('int sum', () => {
    const a  = Series.new({type: new Int32, data: [1, 2, 3]});
    const b  = Series.new({type: new Int32, data: [4, 5, 6]});
    const df = new DataFrame({'a': a, 'b': b});

    expect([...df.sum()]).toEqual([6n, 15n]);
  });

  test('float sum', () => {
    const a  = Series.new({type: new Float32, data: [1, 2.5]});
    const b  = Series.new({type: new Float32, data: [3, 4.5]});
    const df = new DataFrame({'a': a, 'b': b});

    expect([...df.sum()]).toEqual([3.5, 7.5]);
  });

  test('empty dataframe', () => {
    const df = new DataFrame({
      'a': Series.new({type: new Float32, data: []}),
      'b': Series.new({type: new Float32, data: []})
    });
    expect([...df.sum()]).toEqual([null, null]);
  });

  test('skip na is false', () => {
    const df = new DataFrame({'a': Series.new([NaN, 1.5, NaN]), 'b': Series.new([4.5, 5.5, 6.5])});
    expect([...df.sum(undefined, false)]).toEqual([NaN, 16.5]);
  });

  test('skip true is true', () => {
    const df = new DataFrame({'a': Series.new([NaN, 1.5, NaN]), 'b': Series.new([4.5, 5.5, 6.5])});
    expect([...df.sum(undefined, true)]).toEqual([1.5, 16.5]);
  });

  test('subset', () => {
    const df = new DataFrame({'a': Series.new([1, 2, 3]), 'b': Series.new([4.5, 5.5, 6.5])});
    expect([...df.sum(['a'])]).toEqual([6]);
  });

  test('subset and skip na is false', () => {
    const df = new DataFrame({'a': Series.new([NaN, 1.5, NaN]), 'b': Series.new([4.5, 5.5, 6.5])});
    expect([...df.sum(['a'], false)]).toEqual([NaN]);
  });

  test('subset contains incompatiable types', () => {
    const a  = Series.new({type: new Float32, data: [1, 2.5]});
    const b  = Series.new({type: new Float32, data: [3, 4.5]});
    const c  = Series.new({type: new Int32, data: [1, 2]});
    const df = new DataFrame({'a': a, 'b': b, 'c': c});
    expect(() => {
      const result = df.sum(['b', 'c']);
      verifySumResultType(result);
    }).toThrow();
  });

  test('throws if dataframe contains incompatiable types', () => {
    const df = new DataFrame({'a': Series.new(['foo', 'bar']), 'b': Series.new([4.5, 5.5])});
    expect(() => {
      const result = df.sum();
      verifySumResultType(result);
    }).toThrow();

    const df2 = new DataFrame({'a': Series.new([false, true])});
    expect(() => {
      const result = df2.sum();
      verifySumResultType(result);
    }).toThrow();
  });

  test('throws if dataframe contains float and int types', () => {
    const a  = Series.new({type: new Int32, data: [1, 2]});
    const b  = Series.new({type: new Float32, data: [1.5, 2.5]});
    const df = new DataFrame({'a': a, 'b': b});
    expect(() => {
      const result = df.sum();
      verifySumResultType(result);
    }).toThrow();
  });

  // Typescript does not allow us to throw a compile-time error if
  // the return type of `sum()` is `never`.
  // Instead, let's just verify the result is `never` and throw accordingly.
  function verifySumResultType(_: never) { throw new Error(_); }
});
