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

import '@rapidsai/cudf/test/jest-extensions';

import {setDefaultAllocator} from '@rapidsai/cuda';
import {Float32, Float64, FloatingPoint} from '@rapidsai/cudf';
import {Quadtree} from '@rapidsai/cuspatial';
import {DeviceBuffer} from '@rapidsai/rmm';

import {testPoints, testPolygons, testPolylines} from './utils';

setDefaultAllocator((byteLength: number) => new DeviceBuffer(byteLength));

const floatingPointTypes = [
  ['Float32', new Float32],  //
  ['Float64', new Float64]
] as [string, FloatingPoint][];

describe('Quadtree', () => {
  test.each(floatingPointTypes)(
    '`new` constructs a quadtree from points and a bounding box (%s)', (_, type) => {
      const points   = testPoints().castAll(type);
      const quadtree = Quadtree.new({
        x: points.get('x'),
        y: points.get('y'),
        xMin: 0,
        xMax: 8,
        yMin: 0,
        yMax: 8,
        scale: 1,
        maxDepth: 3,
        minSize: 12,
      });

      expect(quadtree.key.data.toArray())
        .toEqualTypedArray(new Uint32Array([0, 1, 2, 0, 1, 3, 4, 7, 5, 6, 13, 14, 28, 31]));
      expect(quadtree.level.data.toArray())
        .toEqualTypedArray(new Uint8Array([0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]));
      expect(quadtree.isQuad.data.toArray())
        .toEqualTypedArray(new Uint8ClampedArray([1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0]));
      expect(quadtree.length.data.toArray())
        .toEqualTypedArray(new Uint32Array([3, 2, 11, 7, 2, 2, 9, 2, 9, 7, 5, 8, 8, 7]));
      expect(quadtree.offset.data.toArray())
        .toEqualTypedArray(new Uint32Array([3, 6, 60, 0, 8, 10, 36, 12, 7, 16, 23, 28, 45, 53]));

      const remapped = points.gather(quadtree.keyMap);

      expect(quadtree.pointX.data.toArray()).toEqualTypedArray(remapped.get('x').data.toArray());
      expect(quadtree.pointY.data.toArray()).toEqualTypedArray(remapped.get('y').data.toArray());
    });

  test(`point in polygon`, () => {
    const points   = testPoints();
    const quadtree = Quadtree.new({
      x: points.get('x'),
      y: points.get('y'),
      xMin: 0,
      xMax: 8,
      yMin: 0,
      yMax: 8,
      scale: 1,
      maxDepth: 3,
      minSize: 12,
    });

    const polygonAndPointIdxs = quadtree.pointInPolygon(testPolygons());
    expect(polygonAndPointIdxs.get('polygon_index').data.toArray())
      .toEqualTypedArray(
        new Uint32Array([3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 3]));

    expect(polygonAndPointIdxs.get('point_index').data.toArray())
      .toEqualTypedArray(new Uint32Array(
        [28, 29, 30, 31, 32, 33, 34, 35, 45, 46, 47, 48, 49, 50, 51, 52, 54, 62, 60]));
  });

  test(`point to nearest polyline`, () => {
    const points   = testPoints();
    const quadtree = Quadtree.new({
      x: points.get('x'),
      y: points.get('y'),
      xMin: 0,
      xMax: 8,
      yMin: 0,
      yMax: 8,
      scale: 1,
      maxDepth: 3,
      minSize: 12,
    });

    const polylinePointPairsAndDistances = quadtree.pointToNearestPolyline(testPolylines(), 2);

    expect(polylinePointPairsAndDistances.get('point_index').data.toArray())
      .toEqualTypedArray(new Uint32Array([
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17,
        18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
        36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
        54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70
      ]));

    expect(polylinePointPairsAndDistances.get('polyline_index').data.toArray())
      .toEqualTypedArray(new Uint32Array([
        3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 3, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
      ]));

    expect(polylinePointPairsAndDistances.get('distance').data.toArray())
      .toEqualTypedArray(new Float64Array([
        3.0675562686570932,   2.5594501016565698,  2.9849608928964071,   1.7103652150920774,
        1.8293181280383963,   1.6095070428899729,  1.681412227243898,    2.3838209461314879,
        2.5510398428020409,   1.6612106150272572,  2.0255119347250292,   2.0660867596957564,
        2.005460353737949,    1.8683447535522375,  1.9465658908648766,   2.215180472008103,
        1.7503944159063249,   1.4820166799617225,  1.6769023397521503,   1.6472789467219351,
        1.0005181046076022,   1.7522309916961678,  1.8490738879835735,   1.0018961233717569,
        0.76002760100291122,  0.65931355999132091, 1.2482129257770731,   1.3229005055827028,
        0.28581819228716798,  0.20466187296772376, 0.41061901127492934,  0.56618357460517321,
        0.046292709584059538, 0.16663093663041179, 0.44953247369220306,  0.56675685520587671,
        0.8426949387264755,   1.2851826443010033,  0.7615641155638555,   0.97842040913621187,
        0.91796378078050755,  1.4311654461101424,  0.96461369875795078,  0.66847988653443491,
        0.98348202146010699,  0.66173276971965733, 0.86233789031448094,  0.50195678903916696,
        0.6755886291567379,   0.82530249944765133, 0.46037120394920633,  0.72651648874084795,
        0.52218906793095576,  0.72892093000338909, 0.077921089704128393, 0.26215098141130333,
        0.33153993710577778,  0.71176747526132511, 0.081119666144327182, 0.60516346789266895,
        0.088508309264124049, 1.5127004224070386,  0.38943741327066272,  0.48717099143018805,
        1.1781283344854494,   1.8030436222567465,  1.0769747770485747,   1.181276832710481,
        1.1240715558969043,   1.6379084234284416,  2.1510078772519496
      ]));
  });
});
