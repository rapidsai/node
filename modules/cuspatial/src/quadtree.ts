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
  Bool8,
  Column,
  DataFrame,
  FloatingPoint,
  Series,
  SeriesType,
  Table,
  Uint32,
  Uint8
} from '@nvidia/cudf';

import {createQuadtree} from './addon';

type QuadtreeSchema = {
  /** Uint32 quad node keys */
  key: Uint32,
  /** Uint8 level for each quadtree node */
  level: Uint8,
  /** Boolean indicating whether a node is a quad or leaf */
  isQuad: Bool8,
  /**
   * If this is a non-leaf quadrant (i.e. `isQuad` is `true`), this is the number of children in
   * the non-leaf quadrant.
   *
   * Otherwise this is the number of points contained in the leaf quadrant.
   */
  length: Uint32,
  /**
   * If this is a non-leaf quadrant (i.e. `isQuad` is `true`), this is the position of the non-leaf
   * quadrant's first child.
   *
   * Otherwise this column's value is the position of the leaf quadrant's first point.
   */
  offset: Uint32,
};

export class Quadtree<T extends FloatingPoint> {
  /**
   * @summary Construct a quadtree from a set of points for a given area-of-interest bounding box.
   *
   * @note Swaps `xMin` and `xMax`` if `xMin > xMax`
   * @note Swaps `yMin` and `yMax`` if `yMin > yMax`
   *
   * @param options Object of quadtree options
   * @param options.x Column of x-coordinates for each point
   * @param options.y Column of y-coordinates for each point
   * @param options.xMin The lower-left x-coordinate of the area of interest bounding box
   * @param options.xMax The upper-right x-coordinate of the area of interest bounding box
   * @param options.yMin The lower-left y-coordinate of the area of interest bounding box
   * @param options.yMax The upper-right y-coordinate of the area of interest bounding box
   * @param options.scale Scale to apply to each point's distance from ``(x_min, y_min)``
   * @param options.maxDepth Maximum quadtree depth in range [0, 15)
   * @param options.minSize Minimum number of points for a non-leaf quadtree node
   * @returns Quadtree
   */
  static new<T extends FloatingPoint>(options: {
    x: Series<T>,
    y: Series<T>,
    xMin: number,
    xMax: number,
    yMin: number,
    yMax: number,
    scale: number,
    maxDepth: number,
    minSize: number,
  }) {
    const xs                       = options.x._col;
    const ys                       = options.y._col;
    const maxDepth                 = Math.max(0, Math.min(15, options.maxDepth | 0));
    const [xMin, xMax, yMin, yMax] = [
      Math.min(options.xMin, options.xMax),
      Math.max(options.xMin, options.xMax),
      Math.min(options.yMin, options.yMax),
      Math.max(options.yMin, options.yMax),
    ];
    const scale = Math.max(options.scale,
                           // minimum valid value for the scale based on bbox and max tree depth
                           Math.max(xMax - xMin, yMax - yMin) / ((1 << maxDepth) + 2));
    const {keyMap, names, table} =
      createQuadtree(xs, ys, xMin, xMax, yMin, yMax, scale, maxDepth, options.minSize);
    return new Quadtree(xs, ys, keyMap, new DataFrame({
                          [names[0]]: Series.new(table.getColumnByIndex<Uint32>(0)),
                          [names[1]]: Series.new(table.getColumnByIndex<Uint8>(1)),
                          [names[2]]: Series.new(table.getColumnByIndex<Bool8>(2)),
                          [names[3]]: Series.new(table.getColumnByIndex<Uint32>(3)),
                          [names[4]]: Series.new(table.getColumnByIndex<Uint32>(4)),
                        }));
  }

  protected constructor(pointX: Column<T>,
                        pointY: Column<T>,
                        keyMap: Column<Uint32>,
                        quadtree: DataFrame<QuadtreeSchema>) {
    this._x        = pointX;
    this._y        = pointY;
    this._keyMap   = keyMap;
    this._quadtree = quadtree;
  }

  /**
   * @summary The x-coordinates for each point used to construct the Quadtree.
   */
  private readonly _x: Column<T>;
  /**
   * @summary The y-coordinates for each point used to construct the Quadtree.
   */
  private readonly _y: Column<T>;

  /**
   * @summary A Uint32 Series of sorted keys to original point indices.
   */
  private readonly _keyMap: Column<Uint32>;

  /**
   * @summary A complete quadtree for the set of input points.
   */
  private readonly _quadtree: DataFrame<QuadtreeSchema>;

  /**
   * @summary A Uint32 Series of quadtree node keys.
   */
  public get key() { return this._quadtree.get('key'); }

  /**
   * @summary A Uint8 Series of the level for each quadtree node.
   */
  public get level() { return this._quadtree.get('level'); }

  /**
   * @summary Boolean indicating whether a node is a quad or leaf.
   */
  public get isQuad() { return this._quadtree.get('isQuad'); }

  /**
   * @summary The number of children or points in each quadrant or leaf node.
   *
   * If this is a non-leaf quadrant (i.e. `isQuad` is `true`), this is the number of children in
   * the non-leaf quadrant.
   *
   * Otherwise this is the number of points contained in the leaf quadrant.
   */
  public get length() { return this._quadtree.get('length'); }

  /**
   * @summary The position of the first child or point in each quadrant or leaf node.
   *
   * If this is a non-leaf quadrant (i.e. `isQuad` is `true`), this is the position of the non-leaf
   * quadrant's first child.
   *
   * Otherwise this column's value is the position of the leaf quadrant's first point.
   */
  public get offset() { return this._quadtree.get('offset'); }

  /**
   * @summary A Uint32 Series mapping each original point index to its sorted z-order in the
   * Quadtree.
   */
  public get keyMap() { return Series.new(this._keyMap); }

  /**
   * @summary A point x-coordinates in the order they appear in the Quadtree.
   */
  public get x(): SeriesType<T> { return Series.new(this._x.gather(this._keyMap)); }

  /**
   * @summary A point y-coordinates in the order they appear in the Quadtree.
   */
  public get y(): SeriesType<T> { return Series.new(this._y.gather(this._keyMap)); }

  /**
   * @summary point x and y-coordinates in the order they appear in the Quadtree.
   */
  public get points() {
    const remap = new Table({columns: [this._x, this._y]}).gather(this._keyMap);
    return new DataFrame({
      x: Series.new(remap.getColumnByIndex<T>(0)),
      y: Series.new(remap.getColumnByIndex<T>(1)),
    });
  }
}
