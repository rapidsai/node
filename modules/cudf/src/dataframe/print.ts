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

import {DataFrame} from '../data_frame';
import {Series} from '../series';
import {Int32, Utf8String} from '../types/dtypes';
import {TypeMap} from '../types/mappings';

export type DisplayOptions = {
  /**
   * The maximum number of columns to display before truncation. Truncation may also occur
   * if the total printed width in characters exceeds `width`
   * (default: 20)
   *
   */
  maxColumns?: number,

  /**
   * The maximum width in characters of a column when printing a DataFrame. When the column *
   * overflows, a "..." placeholder is embedded in the output. A value of zero means unlimited.
   * (default: 50)
   */
  maxColWidth?: number,

  /**
   * The maximum number of rows to output when printing a DataFrame.  A value of zero means
   * unlimited. (default: 60)
   */
  maxRows?: number,

  /**
   * Width of the display in characters. (default: 80)
   */
  width?: number,

}

export class DataFrameFormatter<T extends TypeMap> {
  private frame: DataFrame;
  private maxColumns: number;
  private maxColWidth: number;
  private maxRows: number;
  private width: number;
  private htrunc: boolean;
  private vtrunc: boolean;

  constructor(options: DisplayOptions, frame: DataFrame<T>) {
    this.maxColumns  = options.maxColumns ?? 20;
    this.maxColWidth = options.maxColWidth ?? 50;
    this.maxRows     = options.maxRows ?? 60;
    this.width       = options.width ?? 80;
    this.htrunc      = false;
    this.vtrunc      = false;

    let tmp = this._preprocess(frame);
    while (this._totalWidth(tmp) > this.width) {
      this.htrunc = true;
      const names = [...tmp.names];
      const N     = Math.ceil((names.length - 1) / 2);
      tmp         = tmp.drop([names[N]]);
    }
    this.frame = tmp;
  }

  private _preprocess(frame: DataFrame<T>) {
    let tmp: DataFrame = frame;

    if (this.maxColumns > 0 && tmp.numColumns > this.maxColumns) {
      this.htrunc       = true;
      const half        = this.maxColumns / 2;
      const first       = Math.ceil(half);
      const last        = this.maxColumns - first;
      const last_names  = last == 0 ? [] : tmp.names.slice(-last);
      const trunc_names = [...tmp.names.slice(0, first), ...last_names];
      tmp               = tmp.select(trunc_names);
    }

    if (this.maxRows > 0 && tmp.numRows > this.maxRows - 1) {
      this.vtrunc = true;
      const half  = (this.maxRows - 1) / 2;
      const first = Math.ceil(half);
      const last  = this.maxRows - first - 1;
      tmp         = tmp.head(first).concat(tmp.tail(last));
    }
    return tmp.castAll(new Utf8String).replaceNulls('null');
  }

  render(): string {
    const headers: string[] = [...this.frame.names];
    const widths: number[]  = this._computeWidths(this.frame, headers);
    const lines             = [];

    lines.push(this._formatFields(headers, widths, '   '));

    for (let i = 0; i < this.frame.numRows; ++i) {
      const fields = headers.map((name: string) => this.frame.get(name).getValue(i));
      lines.push(this._formatFields(fields, widths));
    }

    if (this.vtrunc) {
      const N    = 1 + Math.floor((lines.length - 1) / 2);  // 1+ for header row
      const dots = Array(headers.length).fill('...');
      lines.splice(N, 1, this._formatFields(dots, widths));
    }
    lines.push('');

    return lines.join('\n');
  }

  private _totalWidth(frame: DataFrame) {
    const widths = this._computeWidths(frame, [...frame.names]);
    return widths.reduce((x, y) => x + y, 0) + widths.length;
  }

  private _computeWidths(frame: DataFrame, headers: string[]) {
    const inds   = Series.sequence({type: new Int32, size: frame.numRows, init: 0});
    const widths = [];
    for (const name of headers) {
      const lengths   = frame.get(name).len();
      const truncated = lengths.scatter(3, inds.gather(lengths.gt(this.maxColWidth)));  // '...'
      widths.push(Math.max(name.length, truncated.max()));
    }

    return widths;
  }

  _formatFields(rawFields: string[], rawWidths: number[], placeholder = '...') {
    const fields = [...rawFields];
    const widths = [...rawWidths];
    if (this.htrunc) {
      const N = Math.ceil(rawFields.length / 2);
      fields.splice(N, 0, placeholder);
      widths.splice(N, 0, 3);
    }

    const truncated = fields.map((val: string) => val.length > this.maxColWidth ? '...' : val);

    return truncated.map((val: string, i: number) => val.padStart(widths[i], ' ')).join(' ');
  }
}
