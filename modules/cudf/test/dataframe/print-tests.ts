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

import {DataFrame, Int32, Series} from '@rapidsai/cudf';

const a = Series.new([0, 1, 2, 3, 4, 5, 6]);
const b = Series.new([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.61234567]);
const c = Series.new([
  'foo',
  null,
  null,
  'bar',
  'baz',
  null,
  '123456789012345678901234567890123456789012345678901234567890'
]);
const d = a.log();

describe('dataframe.toString', () => {
  test('single column', () => {
    expect(new DataFrame({'a': a}).toString())
      .toEqual(
        `\
  a
0.0
1.0
2.0
3.0
4.0
5.0
6.0
`);

    expect(new DataFrame({'b': b}).toString())
      .toEqual(
        `\
         b
       0.0
       1.1
       2.2
       3.3
       4.4
       5.5
6.61234567
`);
  });

  test('exceeds maxColWidth', () => {
    expect(new DataFrame({'c': c}).toString())
      .toEqual(
        `\
   c
 foo
null
null
 bar
 baz
null
 ...
`);
  });

  test('maxColWidth', () => {
    expect(new DataFrame({'b': b, 'c': c}).toString({maxColWidth: 70}))
      .toEqual(
        `\
         b                                                            c
       0.0                                                          foo
       1.1                                                         null
       2.2                                                         null
       3.3                                                          bar
       4.4                                                          baz
       5.5                                                         null
6.61234567 123456789012345678901234567890123456789012345678901234567890
`);
  });

  test('exceeds width', () => {
    expect(new DataFrame({'a': a, 'b': b, 'c': c, 'd': d}).toString({maxColWidth: 70}))
      .toEqual(
        `\
  a          b               d
0.0        0.0 ...        -Inf
1.0        1.1 ...         0.0
2.0        2.2 ... 0.693147181
3.0        3.3 ... 1.098612289
4.0        4.4 ... 1.386294361
5.0        5.5 ... 1.609437912
6.0 6.61234567 ... 1.791759469
`);
  });

  test('maxRows', () => {
    // 3
    expect(new DataFrame({'a': a, 'b': b, 'c': c, 'd': d}).toString({maxRows: 3}))
      .toEqual(
        `\
  a          b   c           d
0.0        0.0 foo        -Inf
...        ... ...         ...
`);

    // 4
    expect(new DataFrame({'a': a, 'b': b, 'c': c, 'd': d}).toString({maxRows: 4}))
      .toEqual(
        `\
  a          b    c           d
0.0        0.0  foo        -Inf
...        ...  ...         ...
6.0 6.61234567  ... 1.791759469
`);

    // 5
    expect(new DataFrame({'a': a, 'b': b, 'c': c, 'd': d}).toString({maxRows: 5}))
      .toEqual(
        `\
  a          b    c           d
0.0        0.0  foo        -Inf
1.0        1.1 null         0.0
...        ...  ...         ...
6.0 6.61234567  ... 1.791759469
`);

    // 6
    expect(new DataFrame({'a': a, 'b': b, 'c': c, 'd': d}).toString({maxRows: 6}))
      .toEqual(
        `\
  a          b    c           d
0.0        0.0  foo        -Inf
1.0        1.1 null         0.0
...        ...  ...         ...
5.0        5.5 null 1.609437912
6.0 6.61234567  ... 1.791759469
`);

    // 7
    expect(new DataFrame({'a': a, 'b': b, 'c': c, 'd': d}).toString({maxRows: 7}))
      .toEqual(
        `\
  a          b    c           d
0.0        0.0  foo        -Inf
1.0        1.1 null         0.0
2.0        2.2 null 0.693147181
...        ...  ...         ...
5.0        5.5 null 1.609437912
6.0 6.61234567  ... 1.791759469
`);

    // 8
    expect(new DataFrame({'a': a, 'b': b, 'c': c, 'd': d}).toString({maxRows: 8}))
      .toEqual(
        `\
  a          b    c           d
0.0        0.0  foo        -Inf
1.0        1.1 null         0.0
2.0        2.2 null 0.693147181
3.0        3.3  bar 1.098612289
4.0        4.4  baz 1.386294361
5.0        5.5 null 1.609437912
6.0 6.61234567  ... 1.791759469
`);
    // 9 (+1)
    expect(new DataFrame({'a': a, 'b': b, 'c': c, 'd': d}).toString({maxRows: 9}))
      .toEqual(
        `\
  a          b    c           d
0.0        0.0  foo        -Inf
1.0        1.1 null         0.0
2.0        2.2 null 0.693147181
3.0        3.3  bar 1.098612289
4.0        4.4  baz 1.386294361
5.0        5.5 null 1.609437912
6.0 6.61234567  ... 1.791759469
`);
  });

  test('maxColumns', () => {
    const df = new DataFrame({'a': a, 'b': b, 'c': c, 'd': d});

    // 1
    expect(df.toString({maxColumns: 1}))
      // editor trims significant trailing whitespace in a template string
      .toEqual('  a    \n0.0 ...\n1.0 ...\n2.0 ...\n3.0 ...\n4.0 ...\n5.0 ...\n6.0 ...\n');

    // 2
    expect(df.toString({maxColumns: 2}))
      .toEqual(
        `\
  a               d
0.0 ...        -Inf
1.0 ...         0.0
2.0 ... 0.693147181
3.0 ... 1.098612289
4.0 ... 1.386294361
5.0 ... 1.609437912
6.0 ... 1.791759469
`);

    // 3
    expect(df.toString({maxColumns: 3}))
      .toEqual(
        `\
  a          b               d
0.0        0.0 ...        -Inf
1.0        1.1 ...         0.0
2.0        2.2 ... 0.693147181
3.0        3.3 ... 1.098612289
4.0        4.4 ... 1.386294361
5.0        5.5 ... 1.609437912
6.0 6.61234567 ... 1.791759469
`);

    // 4
    expect(df.toString({maxColumns: 4}))
      .toEqual(
        `\
  a          b    c           d
0.0        0.0  foo        -Inf
1.0        1.1 null         0.0
2.0        2.2 null 0.693147181
3.0        3.3  bar 1.098612289
4.0        4.4  baz 1.386294361
5.0        5.5 null 1.609437912
6.0 6.61234567  ... 1.791759469
`);

    // 5 (+1)
    expect(df.toString({maxColumns: 5}))
      .toEqual(
        `\
  a          b    c           d
0.0        0.0  foo        -Inf
1.0        1.1 null         0.0
2.0        2.2 null 0.693147181
3.0        3.3  bar 1.098612289
4.0        4.4  baz 1.386294361
5.0        5.5 null 1.609437912
6.0 6.61234567  ... 1.791759469
`);
  });

  test('exceeds maxRows and maxCols', () => {
    const df = new DataFrame({'a': a, 'b': b, 'c': c, 'd': d});
    expect(df.toString({maxColumns: 3, maxRows: 4}))
      .toEqual(
        `\
  a          b               d
0.0        0.0 ...        -Inf
...        ... ...         ...
6.0 6.61234567 ... 1.791759469
`);
  });

  test('maxRows=0', () => {
    const a  = Series.sequence({type: new Int32, size: 5000, step: 1, init: 0});
    const df = new DataFrame({'a': a});
    expect(df.toString({maxRows: 0}).split('\n').length)
      .toEqual(5002);  // header + trailing newline
  });

  test('maxColumns=0', () => {
    const a         = Series.new([0, 1]);
    const cols: any = {};
    for (let i = 0; i < 30; ++i) { cols[`a${i}`] = a; }
    const df = new DataFrame(cols);
    expect(df.toString({maxColumns: 30, width: 200}))
      .toEqual(
        `\
 a0  a1  a2  a3  a4  a5  a6  a7  a8  a9 a10 a11 a12 a13 a14 a15 a16 a17 a18 a19 a20 a21 a22 a23 a24 a25 a26 a27 a28 a29
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
`);
  });
});
