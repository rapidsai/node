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
  Int32,
  Series,
  Table,
  Uint32,
  Uint8
} from '@nvidia/cudf';
import {MemoryResource} from '@nvidia/rmm';

import {
  createQuadtree,
  findPointsInPolygons,
  findPolylineNearestToEachPoint,
  findQuadtreeAndBoundingBoxIntersections
} from './addon';
import {
  BoundingBoxes,
  Coords,
  polygonBoundingBoxes,
  Polygons,
  polylineBoundingBoxes,
  Polylines
} from './geometry';

type QuadtreeSchema = {
  /** Uint32 quad node keys */
  key: Uint32,
  /** Uint8 level for each quadtree node */
  level: Uint8,
  /** Boolean indicating whether a node is a quad or leaf */
  is_quad: Bool8,
  /**
   * If this is a non-leaf quadrant (i.e. `is_quad` is `true`), this is the number of children in
   * the non-leaf quadrant.
   *
   * Otherwise this is the number of points contained in the leaf quadrant.
   */
  length: Uint32,
  /**
   * If this is a non-leaf quadrant (i.e. `is_quad` is `true`), this is the position of the non-leaf
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
   * @param options.memoryResource Optional resource to use for output device memory allocations.
   * @returns Quadtree
   */
  static new<T extends Coords>(options: {
    x: T,
    y: T,
    xMin: number,
    xMax: number,
    yMin: number,
    yMax: number,
    scale: number,
    maxDepth: number,
    minSize: number,
    memoryResource?: MemoryResource
  }) {
    const x                                         = options.x._col;
    const y                                         = options.y._col;
    const minSize                                   = Math.max(1, options.minSize || 0);
    const {xMin, xMax, yMin, yMax, scale, maxDepth} = normalizeQuadtreeOptions(options);
    const {keyMap, names, table}                    = createQuadtree<T['type']>(
      x, y, xMin, xMax, yMin, yMax, scale, maxDepth, minSize, options.memoryResource);
    return new Quadtree<T['type']>({
      x,
      y,
      keyMap,
      xMin,
      xMax,
      yMin,
      yMax,
      scale,
      maxDepth,
      minSize,
      quadtree: new DataFrame({
        [names[0]]: Series.new(table.getColumnByIndex<Uint32>(0)),
        [names[1]]: Series.new(table.getColumnByIndex<Uint8>(1)),
        [names[2]]: Series.new(table.getColumnByIndex<Bool8>(2)),
        [names[3]]: Series.new(table.getColumnByIndex<Uint32>(3)),
        [names[4]]: Series.new(table.getColumnByIndex<Uint32>(4)),
      })
    });
  }

  protected constructor(options: {
    x: Column<T>,
    y: Column<T>,
    xMin: number,
    xMax: number,
    yMin: number,
    yMax: number,
    scale: number,
    maxDepth: number,
    minSize: number,
    keyMap: Column<Uint32>,
    quadtree: DataFrame<QuadtreeSchema>
  }) {
    this._x        = options.x;
    this._y        = options.y;
    this.xMin      = options.xMin;
    this.xMax      = options.xMax;
    this.yMin      = options.yMin;
    this.yMax      = options.yMax;
    this.scale     = options.scale;
    this.maxDepth  = options.maxDepth;
    this.minSize   = options.minSize;
    this._keyMap   = options.keyMap;
    this._quadtree = options.quadtree;
  }

  /** @summary The x-coordinates for each point used to construct the Quadtree. */
  protected readonly _x: Column<T>;

  /** @summary The y-coordinates for each point used to construct the Quadtree. */
  protected readonly _y: Column<T>;

  /** @summary `xMin` used to construct the Quadtree. */
  public readonly xMin: number;

  /** @summary `xMax` used to construct the Quadtree. */
  public readonly xMax: number;

  /** @summary `yMin` used to construct the Quadtree. */
  public readonly yMin: number;

  /** @summary `yMax` used to construct the Quadtree. */
  public readonly yMax: number;

  /** @summary `scale` used to construct the Quadtree. */
  public readonly scale: number;

  /** @summary `maxDepth` used to construct the Quadtree. * */
  public readonly maxDepth: number;

  /** @summary `minSize` used to construct the Quadtree. */
  public readonly minSize: number;

  /** @summary A Uint32 Series of sorted keys to original point indices. */
  protected readonly _keyMap: Column<Uint32>;

  /** @summary A complete quadtree for the set of input points. */
  protected readonly _quadtree: DataFrame<QuadtreeSchema>;

  /** @summary x-coordinate for each point in their original order. */
  public get x(): Series<T> { return Series.new(this._x); }

  /** @summary y-coordinate for each point in their original order. */
  public get y(): Series<T> { return Series.new(this._y); }

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
  public get isQuad() { return this._quadtree.get('is_quad'); }

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
   * @summary A Uint32 Series mapping each original point index to its sorted position in the
   * Quadtree.
   */
  public get keyMap() { return Series.new(this._keyMap); }

  /**
   * @summary Point x-coordinates in the sorted order they appear in the Quadtree.
   */
  public get pointX(): Series<T> { return Series.new(this._x.gather(this._keyMap)); }

  /**
   * @summary Point y-coordinates in the sorted order they appear in the Quadtree.
   */
  public get pointY(): Series<T> { return Series.new(this._y.gather(this._keyMap)); }

  /**
   * @summary Point x and y-coordinates in the sorted order they appear in the Quadtree.
   */
  public get points() {
    const remap = new Table({columns: [this._x, this._y]}).gather(this._keyMap);
    return new DataFrame({
      x: Series.new(remap.getColumnByIndex<T>(0)),
      y: Series.new(remap.getColumnByIndex<T>(1)),
    });
  }

  /** @ignore */
  public asTable() { return this._quadtree.asTable(); }

  /**
   * @summary Find the subset of the given polygons that contain points in the Quadtree.
   * @param polygons Series of Polygons to test.
   * @param memoryResource Optional resource used to allocate the output device memory.
   * @returns Series of each polygon that contains any points
   */
  public polygonsWithPoints<R extends Polygons<T>>(polygons: R, memoryResource?: MemoryResource) {
    return polygons.gather(this.pointInPolygon(polygons, memoryResource).get('polygon_index'));
  }

  /**
   * @summary Find the subset of points in the Quadtree contained by the given polygons.
   * @param polygons Series of Polygons to test.
   * @param memoryResource Optional resource used to allocate the output device memory.
   * @returns DataFrame x and y-coordinates of each found point
   */
  public pointsInPolygons<R extends Polygons<T>>(polygons: R, memoryResource?: MemoryResource) {
    return new DataFrame({x: this.x, y: this.y})
      .gather(this.pointInPolygon(polygons, memoryResource).get('point_index'));
  }

  /**
   * @summary Find the subset of points in the Quadtree contained by the given polygons.
   * @param polygons Series of Polygons to test.
   * @param memoryResource Optional resource used to allocate the output device memory.
   * @returns DataFrame Indices for each intersecting point and polygon pair.
   */
  public pointInPolygon<R extends Polygons<T>>(polygons: R, memoryResource?: MemoryResource) {
    const intersections =
      this.spatialJoin(polygonBoundingBoxes(polygons, memoryResource), memoryResource);
    const rings          = polygons.elements;
    const polygonPointX  = rings.elements.getChild('x');
    const polygonPointY  = rings.elements.getChild('y');
    const {names, table} = findPointsInPolygons(intersections.asTable(),
                                                this._quadtree.asTable(),
                                                this._keyMap,
                                                this._x,
                                                this._y,
                                                offsetsMinus1(polygons.offsets),
                                                offsetsMinus1(rings.offsets),
                                                polygonPointX._col as Column<T>,
                                                polygonPointY._col as Column<T>,
                                                memoryResource);
    return new DataFrame({
      [names[0]]: Series.new(table.getColumnByIndex<Uint32>(0)),
      [names[1]]: Series.new(table.getColumnByIndex<Uint32>(1)),
    });
  }

  /**
   * @summary Find a subset of points nearest to each given polyline.
   * @param lines Series of Polylines to test.
   * @param expansionRadius Radius of each polyline point.
   * @param memoryResource Optional resource used to allocate the output device memory.
   */
  public pointsNearestPolylines<R extends Polylines<T>>(lines: R,
                                                        expansionRadius = 1,
                                                        memoryResource?: MemoryResource) {
    const result = this.pointToNearestPolyline(lines, expansionRadius, memoryResource);
    return new DataFrame({x: this.x, y: this.y}).gather(result.get('point_index'));
  }

  /**
   * @summary Finds the nearest polyline to each point, and computes the distances between each
   * point/polyline pair.
   * @param polylines Series of Polylines to test.
   * @param expansionRadius Radius of each polyline point.
   * @param memoryResource Optional resource used to allocate the output device memory.
   * @returns DataFrame Indices for each point/nearest polyline pair, and distance between them.
   */
  public pointToNearestPolyline<R extends Polylines<T>>(polylines: R,
                                                        expansionRadius = 1,
                                                        memoryResource?: MemoryResource) {
    const intersections = this.spatialJoin(
      polylineBoundingBoxes(polylines, expansionRadius, memoryResource), memoryResource);
    const polylinePointX = polylines.elements.getChild('x');
    const polylinePointY = polylines.elements.getChild('y');
    const {names, table} = findPolylineNearestToEachPoint(intersections.asTable(),
                                                          this._quadtree.asTable(),
                                                          this._keyMap,
                                                          this._x,
                                                          this._y,
                                                          offsetsMinus1(polylines.offsets),
                                                          polylinePointX._col as Column<T>,
                                                          polylinePointY._col as Column<T>,
                                                          memoryResource);
    return new DataFrame({
      [names[0]]: Series.new(table.getColumnByIndex<Uint32>(0)),
      [names[1]]: Series.new(table.getColumnByIndex<Uint32>(1)),
      [names[2]]: Series.new(table.getColumnByIndex<T>(2)),
    });
  }

  /**
   * @summary Search a quadtree for bounding box intersections.
   * @param boundingBoxes Minimum bounding boxes for a set of polygons or polylines.
   * @param memoryResource Optional resource used to allocate the output device memory.
   * @returns DataFrame Indices for each intersecting bounding box and leaf quadrant.
   */
  public spatialJoin(boundingBoxes: BoundingBoxes<T>, memoryResource?: MemoryResource) {
    const {names, table} = findQuadtreeAndBoundingBoxIntersections(this._quadtree.asTable(),
                                                                   boundingBoxes.asTable(),
                                                                   this.xMin,
                                                                   this.xMax,
                                                                   this.yMin,
                                                                   this.yMax,
                                                                   this.scale,
                                                                   this.maxDepth,
                                                                   memoryResource);
    return new DataFrame({
      [names[0]]: Series.new(table.getColumnByIndex<Uint32>(0)),
      [names[1]]: Series.new(table.getColumnByIndex<Uint32>(1)),
    });
  }
}

function normalizeQuadtreeOptions(options: {
  xMin: number,
  xMax: number,
  yMin: number,
  yMax: number,
  scale: number,
  maxDepth: number,
}) {
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
  return {xMin, xMax, yMin, yMax, scale, maxDepth};
}

function offsetsMinus1(offsets: Series<Int32>) {
  return new Column({type: new Int32, data: offsets.data, length: offsets.length - 1});
}
