// Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

import {DataFrame, scope, Series} from '@rapidsai/cudf';

test('basic disposes', () => {
  let test: any = null;

  const result = scope(() => {
    test = Series.sequence({size: 20});
    return Series.sequence({size: 30});
  });

  expect(test._col.disposed).toBe(true);
  expect(result._col.disposed).toBe(false);
});

test('basic disposes promise', async () => {
  let test: any = null;

  // eslint-disable-next-line @typescript-eslint/require-await
  const promise = scope(async () => {
    test = Series.sequence({size: 20});
    return Series.sequence({size: 30});
  });

  const resolver = async () => promise;
  const result              = await resolver();

  expect(test._col.disposed).toBe(true);
  expect(result._col.disposed).toBe(false);
});

test('explicit keep', () => {
  let test: any = null;
  const outer   = Series.sequence({size: 30});

  const result = scope(() => {
    test        = Series.sequence({size: 20});
    const inner = Series.sequence({size: 30});
    return new DataFrame({
      outer: outer,
      inner: inner,
    });
  }, [outer]);

  expect(test._col.disposed).toBe(true);
  expect(result.get('inner')._col.disposed).toBe(false);
  expect(result.get('outer')._col.disposed).toBe(false);
});

test('nested disposes', () => {
  const outer          = Series.sequence({size: 30});
  let test_inner: any  = null;
  let test_middle: any = null;

  const result = scope(() => {
    const middle = Series.sequence({size: 30});
    test_middle  = Series.sequence({size: 20});

    const result = scope(() => {
      test_inner  = Series.sequence({size: 20});
      const inner = Series.sequence({size: 30});
      return new DataFrame({
        inner: inner,
      });
    });

    expect(test_inner._col.disposed).toBe(true);
    expect(test_middle._col.disposed).toBe(false);
    expect(result.get('inner')._col.disposed).toBe(false);
    expect(middle._col.disposed).toBe(false);
    return result.assign({middle: middle});
  });

  expect(test_inner._col.disposed).toBe(true);
  expect(test_middle._col.disposed).toBe(true);
  expect(result.get('middle')._col.disposed).toBe(false);
  expect(result.get('inner')._col.disposed).toBe(false);
  expect(outer._col.disposed).toBe(false);
});
