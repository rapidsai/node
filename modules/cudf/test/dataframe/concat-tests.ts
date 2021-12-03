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
  Categorical,
  DataFrame,
  Float64,
  Int32,
  Series,
  Utf8String,
} from '@rapidsai/cudf';

describe('dataframe.concat', () => {
  test('zero series in common same types', () => {
    const a   = Series.new([1, 2, 3, 4]);
    const b   = Series.new([5, 6, 7, 8]);
    const dfa = new DataFrame({'a': a});
    const dfb = new DataFrame({'b': b});

    const result = dfa.concat(dfb);

    expect([...result.get('a')]).toEqual([...a, null, null, null, null]);
    expect([...result.get('b')]).toEqual([null, null, null, null, ...b]);
  });

  test('zero series in common different types', () => {
    const a   = Series.new([1, 2, 3, 4]);
    const b   = Series.new(['5', '6', '7', '8']);
    const dfa = new DataFrame({'a': a});
    const dfb = new DataFrame({'b': b});

    const result = dfa.concat(dfb);

    expect([...result.get('a')]).toEqual([...a, null, null, null, null]);
    expect([...result.get('b')]).toEqual([null, null, null, null, ...b]);
  });

  test('one Float64 series in common', () => {
    const a   = Series.new([1, 2, 3, 4]);
    const b   = Series.new([5, 6, 7, 8]);
    const c   = Series.new([9, 10, 11, 12]);
    const dfa = new DataFrame({'a': a, 'b': b});
    const dfb = new DataFrame({'b': b, 'c': c});

    const result = dfa.concat(dfb);

    expect([...result.get('a')]).toEqual([...a, null, null, null, null]);
    expect([...result.get('b')]).toEqual([...b, ...b]);
    expect([...result.get('c')]).toEqual([null, null, null, null, ...c]);
  });

  test('two Float64 series in common', () => {
    const a   = Series.new([1, 2, 3, 4]);
    const b   = Series.new([5, 6, 7, 8]);
    const dfa = new DataFrame({'a': a, 'b': b});
    const dfb = new DataFrame({'a': a, 'b': b});

    const result = dfa.concat(dfb);

    expect([...result.get('a')]).toEqual([...a, ...a]);
    expect([...result.get('b')]).toEqual([...b, ...b]);
  });

  test('one String series in common', () => {
    const a   = Series.new([1, 2, 3, 4]);
    const b   = Series.new(['5', '6', '7', '8']);
    const c   = Series.new([9, 10, 11, 12]);
    const dfa = new DataFrame({'a': a, 'b': b});
    const dfb = new DataFrame({'b': b, 'c': c});

    const result = dfa.concat(dfb);

    expect([...result.get('a')]).toEqual([...a, null, null, null, null]);
    expect([...result.get('b')]).toEqual([...b, ...b]);
    expect([...result.get('c')]).toEqual([null, null, null, null, ...c]);
  });

  test('up-casts Int32 to common Float64 dtype', () => {
    const a1  = Series.new([1, 2, 3, 4, 5]).cast(new Int32);
    const a2  = Series.new([6, 7, 8, 9, 10]);
    const df1 = new DataFrame({'a': a1});
    const df2 = new DataFrame({'a': a2});

    const result = df1.concat(df2);

    // Helper function to throw a compile error if `df.concat()` fails
    // up-cast the `(Int32 | Float64)` type union to Float64, e.g.:
    // ```ts
    // expectFloat64(<Int32|Float64>new Float64());
    // ```
    function expectFloat64(type: Float64) { expect(type).toBeInstanceOf(Float64); }

    expectFloat64(result.get('a').type);
    expect([...result.get('a')]).toEqual([...a1, ...a2]);
  });

  test('fails to up-cast between Float64 and String', () => {
    const a1  = Series.new([1, 2, 3, 4, 5]);
    const a2  = Series.new(['6', '7', '8', '9', '10']);
    const df1 = new DataFrame({'a': a1});
    const df2 = new DataFrame({'a': a2});

    expect(() => {
      // This throws a runtime exception because it
      // can't find a common dtype between Float64 and String.
      // Ideally this should cause a compile-time error, but it
      // TS has no generic equivalent of the C #error directive.
      const result = df1.concat(df2);

      // A compilation error does happen when someone tries to use the "a" Column:
      // result.get('a').type; // `Property 'type' does not exist on type 'never'.ts(2339)`

      // For now, we'll just verify that concat returns a DataFrame<{ a: never }>
      verifyConcatResultType(result);

      function verifyConcatResultType(_: never) { return _; }
    }).toThrow();
  });

  test('array of dataframes', () => {
    const a   = Series.new([1, 2, 3, 4]);
    const b   = Series.new([5, 6, 7, 8]);
    const c   = Series.new([9, 10, 11, 12]);
    const dfa = new DataFrame({'a': a, 'b': b});
    const dfb = new DataFrame({'b': b, 'c': c});
    const dfc = new DataFrame({'a': a, 'c': c});

    const result = dfa.concat(dfb, dfc);

    expect([...result.get('a')]).toEqual([...a, null, null, null, null, ...a]);
    expect([...result.get('b')]).toEqual([...b, ...b, null, null, null, null]);
    expect([...result.get('c')]).toEqual([null, null, null, null, ...c, ...c]);
  });

  test('unique mismatched series length', () => {
    const a   = Series.new([1, 2, 3, 4]);
    const b   = Series.new([5, 6, 7, 8, 9]);
    const dfa = new DataFrame({'a': a});
    const dfb = new DataFrame({'b': b});

    const result = dfa.concat(dfb);
    expect([...result.get('a')]).toEqual([1, 2, 3, 4, null, null, null, null, null]);
    expect([...result.get('b')]).toEqual([null, null, null, null, 5, 6, 7, 8, 9]);
  });

  test('overlapping categorical columns', () => {
    const a  = Series.new(new Int32Array([1, 2, 3, 4])).cast(new Categorical(new Utf8String));
    const aa = Series.new(new Int32Array([5, 6, 1, 3, 6, 5])).cast(new Categorical(new Utf8String));
    const dfa = new DataFrame({'a': a});
    const dfb = new DataFrame({'a': aa});

    const result = dfa.concat(dfb).get('a');
    expect([...result]).toEqual(['1', '2', '3', '4', '5', '6', '1', '3', '6', '5']);
    expect([...result.codes]).toEqual([0, 1, 2, 3, 4, 5, 0, 2, 5, 4]);
    expect([...result.categories]).toEqual(['1', '2', '3', '4', '5', '6']);
  });
});
