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

import {IndexType, Series, Uint32, Uint32Series} from '@rapidsai/cudf';

function getSize(col: Series<IndexType>|number) {
  return col instanceof Series ? col.length : undefined;
}

function validateSeries(col: Series<IndexType>) {
  if (col.min() < 0 || col.max() > 255) {
    throw new RangeError('invalid column, contains values outside of the range [0,255]');
  }
  return col;
}

function hexToInteger(value: string) {
  value = value.replace('#', '0x');

  return parseInt(`${value}ff`, 16);
}
/**
 * Convert 4 individual RGBA Series or values to a RGBA Integer Series to be consumed directly by
 * color buffers
 * @param r Series<IndexType> or number for the color red (valid values [0,255])
 * @param g Series<IndexType> or number for the color red (valid values [0,255])
 * @param b Series<IndexType> or number for the color red (valid values [0,255])
 * @param a Series<IndexType> or number for the color red (valid values [0,255])
 * @param size optional size of the series, if all of the r,g,b,a arguments are numbers. default is
 *   1
 * @returns Uint32Series containing RGBA integer values
 */
export function RGBASeriestoIntSeries(r: Series<IndexType>|number,
                                      g: Series<IndexType>|number,
                                      b: Series<IndexType>|number,
                                      a: Series<IndexType>|number,
                                      size = 1): Uint32Series {
  size = getSize(r) || getSize(g) || getSize(b) || getSize(a) || size;
  r    = r instanceof Series ? validateSeries(r)
                             : Series.sequence({type: new Uint32, size: size, init: r, step: 0});
  g    = g instanceof Series ? validateSeries(g)
                             : Series.sequence({type: new Uint32, size: size, init: g, step: 0});
  b    = b instanceof Series ? validateSeries(b)
                             : Series.sequence({type: new Uint32, size: size, init: b, step: 0});
  a    = a instanceof Series ? validateSeries(a)
                             : Series.sequence({type: new Uint32, size: size, init: a, step: 0});

  return r.shiftLeft(BigInt(24))
    .bitwiseOr(g.shiftLeft(BigInt(16)))
    .bitwiseOr(b.shiftLeft(BigInt(8)))
    .bitwiseOr(a)
    .cast(new Uint32);
}
/**
 * Convert rgba to 32 bit signed integer
 * @param rgba array of rgba value (a value is optional, default is 255)
 * @returns rgba 32 bit signed integer value
 */
export function RGBAtoInt(rgba: number[]): number {
  rgba.forEach((value) => {
    if (value < 0 || value > 255) { throw new RangeError('rgba values expected within [0,255]'); }
  });
  if (rgba.length < 3 || rgba.length > 4) {
    throw new Error('invalid array length, provide values for r,g,b,a(optional)');
  }
  if (rgba.length == 3) { rgba = rgba.concat(255); }  // if only rgb values are provided
  return (rgba[0] << 24 >>> 0) + (rgba[1] << 16 >>> 0) + (rgba[2] << 8 >>> 0) + rgba[3];
}

/**
 * Helper function for quickly creating a cudf color Series based on color bins provided
 * @param values
 * @param domain
 * @param colors
 *    possible input values:
 *      - array of rgba arrays
 *      - array of rgb arrays (a defaults to 255)
 *      - array of 32bit rgba integers
 *      - array of hex color strings
 * @param nullColor
 */
export function mapValuesToColorSeries(
  values: Series<IndexType>,
  domain: number[],
  colors: number[][]|number[]|string[],
  nullColor: number[]|number = [204, 204, 204, 255]): Uint32Series {
  // validate colors and domain lengths
  if (colors.length < 1 || domain.length < 1) {
    throw new Error('colors and domain must be arrays of length 1 or greater');
  }

  // convert RGBA values to Integers accepted by deck gpu buffers
  const nullColorInteger = Array.isArray(nullColor) ? RGBAtoInt(nullColor) : nullColor;
  const colorsInteger: number[] =
    colors.map(value => Array.isArray(value)       ? RGBAtoInt(value)
                        : typeof value == 'string' ? hexToInteger(value)
                                                   : value);
  let colorSeries =
    Series.sequence({type: new Uint32, init: colorsInteger[0], step: 0, size: values.length});
  const colorIndices = Series.sequence({type: values.type, init: 0, step: 1, size: values.length});

  if (domain.length == 1) {
    const boolMask = values.ge(domain[0]);
    const indices  = colorIndices.filter(boolMask);
    colorSeries =
      colorSeries.scatter(colorsInteger[1] || colorsInteger[colors.length - 1], indices);
  } else {
    for (let i = 0; i < domain.length - 1; i++) {
      const boolMask = values.ge(domain[i]).logicalAnd(values.lt(domain[i + 1]));
      const indices  = colorIndices.filter(boolMask);
      colorSeries =
        colorSeries.scatter(colorsInteger[i] || colorsInteger[colors.length - 1], indices);
    }
  }
  // handle nulls
  if (values.countNonNulls() !== values.length) {  // contains null values
    const indices = colorIndices.filter(values.isNull());
    colorSeries   = colorSeries.scatter(nullColorInteger, indices);
  }

  return colorSeries;
}
