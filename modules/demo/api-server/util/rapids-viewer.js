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

const {Bool8, Utf8String, Int32, Int64, DataFrame, Series, Float32, Float64} =
  require('@rapidsai/cudf');

const cuspatial   = require('@rapidsai/cuspatial');
const Prioritizer = require('./prioritizer');
const Budgeter    = require('./budgeter');

module.exports = {
  set_df(df, xAxisName, yAxisName) {
    /*
     * Stores the dataframe and the names of the x and y axes.
     *
     * @param {DataFrame} df
     * @param {String} xAxisName
     * @param {String} yAxisName
     * @return {void}
     */
    this.df                        = df;
    this.xAxisName                 = xAxisName;
    this.yAxisName                 = yAxisName;
    const x                        = this.df.get(this.xAxisName).cast(new Float64);
    const y                        = this.df.get(this.yAxisName).cast(new Float64);
    const [xMin, xMax, yMin, yMax] = [x.min(), x.max(), y.min(), y.max()];
    this.quadtree                  = cuspatial.Quadtree.new(
      {x, y, xMin, xMax, yMin, yMax, scale: -1.0, maxDepth: 15, minSize: 1e5});
    this._viewportCache = new Map();
  },

  _load_viewport_polygon(lb, ub) {
    let cachedViewport = this._viewportCache.get([lb, ub]);
    if (cachedViewport !== undefined) { return cachedViewport; }

    this.lb              = lb;
    this.ub              = ub;
    const pts            = cuspatial.makePoints(Series.new([lb[0], ub[0], ub[0], lb[0], lb[0]]),
                                     Series.new([lb[1], lb[1], ub[1], ub[1], lb[1]]));
    const ring_offset    = [0, 5];
    const polygon_offset = [0, 1];
    const polylines      = cuspatial.makePolylines(pts, ring_offset);
    const polygons       = cuspatial.makePolygons(polylines, polygon_offset);
    this._viewportCache.set([lb, ub], polygons);
    return polygons
  },

  set_viewport(lb, ub) {
    /*
     * Stores the viewport and creates a budgeter on the points in the viewport.
     *
     * @param {Array} lb
     * @param {Array} ub
     * @return {void}
     */
    if (!this.quadtree) { throw new Error('Must set dataframe before setting viewport'); }
    const polygons         = this._load_viewport_polygon(lb, ub);
    const polyPointPairs   = this.quadtree.pointInPolygon(polygons);
    const pointsInViewport = this.quadtree.points.gather(polyPointPairs.get('point_index'));
    this.viewportPoints    = new Budgeter(pointsInViewport);
    const budgetPoints =
      this.viewportPoints.get_n(this.budget ? this.budget : pointsInViewport.numRows);
    this.budgetedPoints = new Prioritizer(budgetPoints);
    this.budgetedPoints.set_priorities(Series.sequence(
      {type: new Int32, size: this.budgetedPoints.points.numRows, init: 1, step: 0}));
  },

  change_budget(budget) {
    /*
     * Changes the budget.
     *
     * @param {Number} budget
     * @return {void}
     */
    this.budget = budget;
    this.viewportPoints.reset();
    const budgetPoints  = this.viewportPoints.get_n(this.budget);
    this.budgetedPoints = new Prioritizer(budgetPoints);
    this.budgetedPoints.set_priorities(Series.sequence(
      {type: new Int32, size: this.budgetedPoints.points.numRows, init: 1, step: 0}));
  },

  _interleave_points(points) {
    let result_col =
      Series.sequence({size: points.numRows * 2, type: new Float32, step: 0, init: 0});
    result_col = result_col.scatter(
      points.get('x'), Series.sequence({size: points.numRows, type: new Int32, step: 2, init: 0}));
    result_col = result_col.scatter(
      points.get('y'), Series.sequence({size: points.numRows, type: new Int32, step: 2, init: 1}));
    return result_col
  },

  get_n(n) {
    /*
     * Samples n points from the quadtree.
     *
     * @param {Number} n
     * @return {Array}
     *  An array of points that were sampled.
     */
    const points      = this.budgetedPoints.get_n(n);
    const interleaved = this._interleave_points(points);
    const result      = new DataFrame({'points_in_polygon': interleaved})
    return result
  }
}
