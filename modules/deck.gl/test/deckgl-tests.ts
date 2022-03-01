// Copyright (c) 2022, NVIDIA CORPORATION.
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

import {Series, Uint32, Uint8} from '@rapidsai/cudf';
import {mapValuesToColorSeries, RGBASeriestoIntSeries, RGBAtoInt} from '@rapidsai/deck.gl';

describe('DeckGL-color-utils', () => {
  test('test RGBAtoInt', () => {
    expect(RGBAtoInt([255, 255, 255, 255])).toEqual(4294967295);
    expect(RGBAtoInt([167, 0, 125, 0])).toEqual(2801827072);
    expect(RGBAtoInt([123, 123, 123, 125])).toEqual(2071690109);
    expect(() => {RGBAtoInt([123, 123, 256, 125])}).toThrow(RangeError);
  });

  test('test RGBASeriesToIntSeries', () => {
    const r    = Series.new({type: new Uint32, data: [255, 167, 123]});
    const g    = Series.new({type: new Uint32, data: [255, 0, 123]});
    const b    = Series.new({type: new Uint32, data: [255, 125, 123]});
    const a    = Series.new({type: new Uint32, data: [255, 0, 125]});
    const badR = Series.new({type: new Uint32, data: [-1, 167, 256]});

    const expectedA = Series.new({type: new Uint32, data: [4294967295, 2801827072, 2071690109]});
    const expectedB = Series.new({type: new Uint32, data: [4294967295, 2801827327, 2071690239]});

    expect([...RGBASeriestoIntSeries(r, g, b, a)]).toEqual([...expectedA]);
    expect([...RGBASeriestoIntSeries(r, g, b, 255)]).toEqual([...expectedB]);
    expect(() => {[...RGBASeriestoIntSeries(badR, g, b, 255)]}).toThrow(RangeError);
  });

  test('test mapValuesToColorSeries', () => {
    const values = Series.new({type: new Uint8, data: [1, 5, 19, 24, 23, 32, 50, null]});
    const domain = [1, 10, 30, 40];
    // Integer colors: 4278190335, 4227793151, 1835007, 16712447, 4278243071
    const colors    = [[255, 0, 0], [251, 255, 0], [0, 27, 255], [0, 255, 2], [255, 0, 206]];
    const colorsHex = ['#ff0000', '#fbff00', '#001bff', '00ff02', 'ff00ce'];

    const resultColors = Series.new({
      type: new Uint32,
      data: [
        4278190335,
        4278190335,
        4227793151,
        4227793151,
        4227793151,
        1835007,
        4278190335,
        3435973887  // default null Color
      ]
    })

    expect([...mapValuesToColorSeries(values, domain, colors)]).toEqual([...resultColors]);
    expect([...mapValuesToColorSeries(values, domain, colorsHex)]).toEqual([...resultColors]);
  })
});
