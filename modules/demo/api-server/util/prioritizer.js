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

const {Series, DataFrame} = require('@rapidsai/cudf');
const Budgeter            = require('./budgeter');

class Prioritizer {
  constructor(points) { this.points = points; }

  set_priorities(priority_array) {
    // Optimization:
    // Get unique priority levels
    // Create one point machine for each priority level.
    this.point_machines = [
      new Budgeter(this.points.filter(priority_array.eq(1))),
      new Budgeter(this.points.filter(priority_array.eq(2))),
      new Budgeter(this.points.filter(priority_array.eq(3))),
      new Budgeter(this.points.filter(priority_array.eq(4))),
    ];
  }

  get_n(budget) {
    if (this.point_machines === undefined) {
      throw new Error('Prioritizer.get_n called before set_priorities');
    }
    // Going to carry the empty dataframe through so it doesn' thave to be initialized.
    let prioritized = undefined;
    for (let i = 0; i < this.point_machines.length; i++) {
      prioritized = this.point_machines[i].get_n(budget);
      if (prioritized.numRows > 0) { return prioritized; }
    }
    // Budget has been emptied case, return empty dataframe
    return prioritized;
  }
}

module.exports = Prioritizer;
