// Copyright (c) 2023, NVIDIA CORPORATION.
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

const {Int32, Series} = require('@rapidsai/cudf');

class Budgeter {
  constructor(points) {
    this.called          = 0;
    this.displayed_count = 0;
    this._last_budget    = null;
    this.points          = points;
  }

  get_n(budget) {
    if (this._last_budget !== null && budget !== this._last_budget) {
      // Budget was changed
      this.called = 0;
    }
    const step = this.points.length / budget;
    const indices =
      Series.sequence({type: new Int32, init: 0, size: budget}).mul(step).add(this.called);
    if (this.called === Math.floor(step)) {
      // Return the final points that weren't previously sent.
      const final_range =
        Series.sequence({type: new Int32, init: this.displayed_count, size: this.points.length});
      this.called++;
      this.displayed_count += final_range.length;
      return this.points.gather(final_range, true).dropNulls();
    } else if (this.called > Math.floor(step)) {
      // All out of points
      this.called++;
      return Series.new([]);
    }
    this.called++;
    this.displayed_count += indices.length;
    return this.points.gather(indices.cast(new Int32));
  }

  get_displayed_count() { return this.displayed_count; }
}

module.exports = Budgeter;
