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

export type Point<T extends FloatingPoint>    = Struct<{x: T, y: T}>;
export type Polyline<T extends FloatingPoint> = List<Point<T>>;
export type Polygon<T extends FloatingPoint>  = List<Polyline<T>>;
export type BoundingBoxes<T extends FloatingPoint> =
  DataFrame<{x_min: T, y_min: T, x_max: T, y_max: T}>;

export function points<T extends FloatingPoint>(x: Series<T>, y: Series<T>): Series<Point<T>> {
  return Series.new<Point<T>>({
    type: new Struct([
      arrow.Field.new('x', x.type as T),
      arrow.Field.new('y', y.type as T),
    ]),
    children: [x, y]
  });
}

export function polylines<T extends FloatingPoint>(offsets: Series<Int32>,
                                                   points: Series<Point<T>>): Series<Polyline<T>> {
  return Series.new({
    children: [offsets, points],
    type: new List(arrow.Field.new('points', <any>points.type as Point<T>)),
  });
}

export function polylineBoundingBoxes<T extends FloatingPoint>(
  polylines: Series<Polyline<T>>, expansionRadius = 1, memoryResource?: MemoryResource):
  BoundingBoxes<T> {
  const points         = polylines.elements;
  const xs             = points.getChild('x');
  const ys             = points.getChild('y');
  const {names, table} = computePolylineBoundingBoxes(polylines.offsets._col,
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

export function polygons<T extends FloatingPoint>(offsets: Series<Int32>,
                                                  rings: Series<Polyline<T>>): Series<Polygon<T>> {
  return Series.new({
    children: [offsets, rings],
    type: new List(arrow.Field.new('rings', <any>rings.type as Polyline<T>)),
  });
}

export function polygonBoundingBoxes<T extends FloatingPoint>(
  polygons: Series<Polygon<T>>, memoryResource?: MemoryResource): BoundingBoxes<T> {
  const rings          = polygons.elements;
  const points         = rings.elements;
  const xs             = points.getChild('x');
  const ys             = points.getChild('y');
  const {names, table} = computePolygonBoundingBoxes(polygons.offsets._col,
                                                     rings.offsets._col,
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
