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

import * as d3 from "d3";

export function generateLegend(container, keys, thresholdScale) {
  // select the svg area
  var svg = d3.select(container)

  // Add one dot in the legend for each name.
  var size = 20
  svg.selectAll('mydots')
    .data(Object.keys(keys))
    .enter()
    .append('rect')
    .attr('x', 10)
    .attr('y',
      function (d, i) { return 10 + i * (size + 5) })  // 100 is where the first dot appears.
    // 25 is the distance between dots
    .attr('width', size)
    .attr('height', size)
    .style('fill',
      function (d) {
        const values = thresholdScale(d);
        return 'rgb(' + values.join(', ') + ')';
      })
    .style('opacity', 0.8)

  // Add one dot in the legend for each name.
  svg.selectAll('mylabels')
    .data(Object.keys(keys))
    .enter()
    .append('text')
    .attr('x', 10 + size * 1.2)
    .attr('y',
      function (d, i) {
        return 10 + i * (size + 5) + (size / 2)
      })  // 100 is where the first dot appears. 25 is the distance between dots
    .style('fill',
      function (d) {
        const values = thresholdScale(d);
        return 'rgb(' + values.join(', ') + ')';
      })
    .text(function (d) { return keys[d] })
    .attr('text-anchor', 'left')
    .style('alignment-baseline', 'middle')
    .style('opacity', 0.8)
}
