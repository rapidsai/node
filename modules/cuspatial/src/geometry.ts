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

import {Column, DataFrame, FloatingPoint, Int32, List, Series, Struct} from '@nvidia/cudf';
import {MemoryResource} from '@nvidia/rmm';
import * as arrow from 'apache-arrow';

import {computePolygonBoundingBoxes, computePolylineBoundingBoxes} from './addon';

export type BoundingBoxes<T extends FloatingPoint> =
  DataFrame<{x_min: T, y_min: T, x_max: T, y_max: T}>;

export type Coord<T extends FloatingPoint = FloatingPoint>  = T;
export type Coords<T extends FloatingPoint = FloatingPoint> = Series<T>;

export type Point<T extends Coord = FloatingPoint>          = Struct<{x: T, y: T}>;
export type Points<T extends FloatingPoint = FloatingPoint> = Series<Point<Coord<T>>>;

export type Polyline<T extends Point = Point>                  = List<T>;
export type Polylines<T extends FloatingPoint = FloatingPoint> = Series<List<Point<T>>>;

export type Polygon<T extends Polyline = Polyline>            = List<T>;
export type Polygons<T extends FloatingPoint = FloatingPoint> = Series<List<Polyline<Point<T>>>>;

export function makePoints<T extends Series<FloatingPoint>>(x: T, y: T): Points<T['type']> {
  return Series.new({
    type: new Struct([
      arrow.Field.new('x', x.type),
      arrow.Field.new('y', y.type),
    ]),
    children: [x, y]
  });
}

export function makePolylines<T extends FloatingPoint>(points: Points<T>,
                                                       offsets: Series<Int32>): Polylines<T> {
  return Series.new({
    children: [offsets, points],
    type: new List(arrow.Field.new('points', points.type)),
  });
}

export function polylineBoundingBoxes<T extends FloatingPoint>(
  polylines: Polylines<T>, expansionRadius = 1, memoryResource?: MemoryResource) {
  const points         = polylines.elements;
  const xs             = points.getChild('x');
  const ys             = points.getChild('y');
  const {names, table} = computePolylineBoundingBoxes<T>(offsetsMinus1(polylines.offsets),
                                                         xs._col as Column<T>,
                                                         ys._col as Column<T>,
                                                         expansionRadius,
                                                         memoryResource);
  return <any>new DataFrame({
    [names[0]]: Series.new(table.getColumnByIndex<T>(0)),
    [names[1]]: Series.new(table.getColumnByIndex<T>(1)),
    [names[2]]: Series.new(table.getColumnByIndex<T>(2)),
    [names[3]]: Series.new(table.getColumnByIndex<T>(3)),
  });
}

export function makePolygons<T extends FloatingPoint>(rings: Polylines<T>,
                                                      offsets: Series<Int32>): Polygons<T> {
  return Series.new({
    children: [offsets, rings],
    type: new List(arrow.Field.new('rings', rings.type)),
  });
}

export function polygonBoundingBoxes<T extends FloatingPoint>(polygons: Polygons<T>,
                                                              memoryResource?: MemoryResource) {
  const rings          = polygons.elements;
  const points         = rings.elements;
  const xs             = points.getChild('x');
  const ys             = points.getChild('y');
  const {names, table} = computePolygonBoundingBoxes<T>(offsetsMinus1(polygons.offsets),
                                                        offsetsMinus1(rings.offsets),
                                                        xs._col as Column<T>,
                                                        ys._col as Column<T>,
                                                        memoryResource);
  return <any>new DataFrame({
    [names[0]]: Series.new(table.getColumnByIndex<T>(0)),
    [names[1]]: Series.new(table.getColumnByIndex<T>(1)),
    [names[2]]: Series.new(table.getColumnByIndex<T>(2)),
    [names[3]]: Series.new(table.getColumnByIndex<T>(3)),
  });
}

function offsetsMinus1(offsets: Series<Int32>) {
  return new Column({type: new Int32, data: offsets.data, length: offsets.length - 1});
}
