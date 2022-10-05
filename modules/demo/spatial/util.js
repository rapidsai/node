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
  DataFrame,
  Float32,
  Int32,
  List,
  Series,
  Struct,
  Uint32,
  Uint8,
} from '@rapidsai/cudf';
import {Quadtree} from '@rapidsai/cuspatial';
import {tableFromIPC} from 'apache-arrow';
import {existsSync} from 'fs';
import {readFile as fsReadFile} from 'fs/promises';
import * as Path from 'path';

import * as loadSpatialDataset from './data';

/**
 * @param {Map<number, [number, number, number, number]>} colorMap
 */
export async function loadPointsInCensusTracts(colorMap) {
  const {tracts, bbox, points, polyIds, levels} =
    await Promise.all([loadTracts(), loadPoints()]).then(([{tracts, bbox}, {quadtree}]) => ({
                                                           tracts,
                                                           bbox,
                                                           ...pointsInCensusTracts(tracts,
                                                                                   quadtree),
                                                         }));
  const colors         = computeColors(tracts, points, colorMap);
  const pointsByLevel  = filterPointsAndColorsByLevel(colors, levels, points, polyIds);
  const tract_polygons = copyPolygonsDtoH(tracts, colorMap);
  const tract_vertices = copyPolygonVerticesDtoH(tracts, colorMap);
  return {
    tracts,
    bbox,
    points,
    polyIds,
    levels,
    colors,
    pointsByLevel,
    tract_polygons,
    tract_vertices,
  };
}

/**
 * @param {Map<number, [number, number, number, number]>} colorMap
 */
export async function loadPointsNearEachCensusTract(colorMap) {
  const {tracts, bbox, points, polyIds, levels} =
    await Promise.all([loadTracts(), loadPoints()]).then(([{tracts, bbox}, {quadtree}]) => ({
                                                           tracts,
                                                           bbox,
                                                           ...pointsNearCensusTracts(tracts,
                                                                                     quadtree),
                                                         }));
  const colors         = computeColors(tracts, points, colorMap);
  const pointsByLevel  = filterPointsAndColorsByLevel(colors, levels, points, polyIds);
  const tract_polygons = copyPolygonsDtoH(tracts, colorMap);
  const tract_vertices = copyPolygonVerticesDtoH(tracts, colorMap);
  return {
    tracts,
    bbox,
    points,
    polyIds,
    levels,
    colors,
    pointsByLevel,
    tract_polygons,
    tract_vertices,
  };
}

export async function loadTracts() {
  console.time(`load geometry Arrow table (${(263).toLocaleString()} polys)`);

  const table = tableFromIPC(await fsReadFile(Path.join(__dirname, 'data', '263_tracts.arrow')));

  console.timeEnd(`load geometry Arrow table (${(263).toLocaleString()} polys)`);

  console.time(`copy census tracts to GPU and compute bbox (${(263).toLocaleString()} polys)`);

  /**
   * @type DataFrame<{ id: Int32, polygon: List<List<Struct<{x: Float32, y: Float32}>>> }>
   */
  const tracts = new DataFrame({
    id: Series.new(table.getChildAt(0)),
    polygon: Series.new(table.getChildAt(1)),
  });

  const [xMin, xMax] = tracts.get('polygon').elements.elements.getChild('x').minmax();
  const [yMin, yMax] = tracts.get('polygon').elements.elements.getChild('y').minmax();

  console.timeEnd(`copy census tracts to GPU and compute bbox (${(263).toLocaleString()} polys)`);

  return {
    tracts,
    bbox: {
      xMin,
      xMax,
      yMin,
      yMax,
      width: xMax - xMin,
      height: yMax - yMin,
    }
  };
}

export async function loadPoints() {
  const table = await loadPointsTable();

  console.time(`copy points to GPU (${(168898952).toLocaleString()} points)`);

  /**
   * @type DataFrame<{ x: Float32, y: Float32 }>
   */
  const points = new DataFrame({
    x: Series.new(table.getChildAt(0)),
    y: Series.new(table.getChildAt(1)),
  });

  console.timeEnd(`copy points to GPU (${(168898952).toLocaleString()} points)`);

  return {points, ...createQuadtree(points)};

  async function loadPointsTable(loadDatasetIfNotFound = true) {
    try {
      console.time(`load points Arrow table (${(168898952).toLocaleString()} points)`);
      const filePath = Path.join(__dirname, 'data', '168898952_points.arrow');
      if (!existsSync(filePath)) { throw new Error(`file not found: "${filePath}"`); }
      const table = tableFromIPC(await fsReadFile(filePath));
      console.timeEnd(`load points Arrow table (${(168898952).toLocaleString()} points)`);
      return table;
    } catch (e) {
      if (loadDatasetIfNotFound) {
        console.error(e);
        console.log('dataset not found, now downloading...');
        return await loadSpatialDataset().then(() => loadPointsTable(false))
      }
      console.error(`
  Point data not found! Run this to download the sample data from AWS S3 (1.3GiB):

  node ${Path.join(__dirname, 'data.js')}
  `);
      process.exit(1);
    }
  }
}

/**
 *
 * @param {DataFrame<{ x: Float32; y: Float32; }>} points
 * @returns
 */
export function createQuadtree(points) {
  console.time(`construct quadtree for ${points.numRows.toLocaleString()} points`);

  const [xMin, xMax]    = points.get('x').minmax();
  const [yMin, yMax]    = points.get('y').minmax();
  const [width, height] = [xMax - xMin, yMax - yMin];
  const scale           = Math.min(width / window.outerWidth, height / window.outerHeight);
  const minSize         = Math.sqrt(points.numRows);
  const quadtree        = Quadtree.new({
    x: points.get('x'),
    y: points.get('y'),
    xMin,
    xMax,
    yMin,
    yMax,
    maxDepth: 15,
    scale,
    minSize,
  });

  console.timeEnd(`construct quadtree for ${points.numRows.toLocaleString()} points`);

  return {
    quadtree,
    bbox: {
      xMin,
      xMax,
      yMin,
      yMax,
      width: xMax - xMin,
      height: yMax - yMin,
    }
  };
}

/**
 * @param {DataFrame<{ id: Int32, polygon: List<List<Struct<{x: Float32, y: Float32}>>> }>} tracts
 * @param {Quadtree<Float32>} quadtree
 */
export function pointsInCensusTracts(tracts, quadtree) {
  // Filter points by which census tract they're in
  console.time(`query for points in census tract polygons (${
    quadtree.keyMap.length.toLocaleString()} points, ${tracts.numRows.toLocaleString()} polygons)`);

  const polyAndPointIdxs = quadtree.pointInPolygon(tracts.get('polygon'));
  const polyIds          = polyAndPointIdxs.get('polygon_index');

  console.timeEnd(`query for points in census tract polygons (${
    quadtree.keyMap.length.toLocaleString()} points, ${tracts.numRows.toLocaleString()} polygons)`);

  console.time(`gather query result points (${polyIds.length.toLocaleString()} points)`);

  const filteredPoints = quadtree.points.gather(polyAndPointIdxs.get('point_index'));
  const filteredLevels = quadtree.level.gather(polyIds);

  console.timeEnd(`gather query result points (${polyIds.length.toLocaleString()} points)`);

  return {points: filteredPoints, levels: filteredLevels, polyIds, tracts, quadtree};
}

/**
 * @param {DataFrame<{ id: Int32, polygon: List<List<Struct<{x: Float32, y: Float32}>>> }>} tracts
 * @param {Quadtree<Float32>} quadtree
 */
export function pointsNearCensusTracts(tracts, quadtree) {
  // Filter points by which census tract they're in
  console.time(`query for points nearest to each census tract boundary (${
    quadtree.keyMap.length.toLocaleString()} points, ${tracts.numRows.toLocaleString()} polygons)`);

  const polyAndPointIdxs = quadtree.pointToNearestPolyline(tracts.get('polygon').elements, 1);
  const polyIds          = polyAndPointIdxs.get('polyline_index');

  console.timeEnd(`query for points nearest to each census tract boundary (${
    quadtree.keyMap.length.toLocaleString()} points, ${tracts.numRows.toLocaleString()} polygons)`);

  console.time(`gather query result points (${polyIds.length.toLocaleString()} points)`);

  const filteredPoints = quadtree.points.gather(polyAndPointIdxs.get('point_index'));
  const filteredLevels = quadtree.level.gather(polyIds);

  console.timeEnd(`gather query result points (${polyIds.length.toLocaleString()} points)`);

  return {points: filteredPoints, levels: filteredLevels, polyIds, tracts, quadtree};
}

/**
 * @param {DataFrame<{ id: Int32, polygon: List<List<Struct<{x: Float32, y: Float32}>>> }>} tracts
 * @param {DataFrame<{ x: Float32; y: Float32; }>} points
 * @param {Map<number, [number, number, number, number]>} colorMap
 */
export function computeColors(tracts, points, colorMap) {
  console.time(`compute colors (${points.numRows.toLocaleString()} points)`);

  const colors = (() => {
    const swizzle = ([r, g, b, a]) => [b, g, r, a];
    const idxs                     = Array.from({length: tracts.numRows}, (_, i) => i);
    const palette                  = idxs.map((i) => swizzle(colorMap.get(i))).flat();
    return Series.new(new Uint8Array(palette)).view(new Uint32);
  })();

  console.timeEnd(`compute colors (${points.numRows.toLocaleString()} points)`);

  return colors;
}

/**
 *
 * @param {Series<Uint32>} colors
 * @param {Series<Uint8>} levels
 * @param {DataFrame<{x:Float32, y:Float32}>} points
 * @param {Series<Uint32>} polyIds
 * @returns
 */
export function filterPointsAndColorsByLevel(colors, levels, points, polyIds) {
  const [minLevel, maxLevel] = levels.minmax().map(Number);
  // console.log({minLevel, maxLevel});

  console.time(`filter points and colors by level (${points.numRows.toLocaleString()} points)`);

  const pointsByLevel = Array.from({length: (maxLevel - minLevel) + 1}, (_, i) => {
    const mask      = levels.eq(minLevel + i);
    const lvlIds    = polyIds.filter(mask);
    const lvlPoints = points.filter(mask);
    const lvlColors = colors.gather(lvlIds);
    return {
      level: minLevel + i,
      numNodes: mask.sum(),
      data: getPointsInBatches(lvlPoints, lvlColors, lvlIds)
    };
  });

  console.timeEnd(`filter points and colors by level (${points.numRows.toLocaleString()} points)`);

  return pointsByLevel;
}

/**
 * @param {DataFrame<{ id: Int32, polygon: List<List<Struct<{x: Float32, y: Float32}>>> }>} tracts
 * @param {Map<number, [number, number, number, number]>} colorMap
 */
export function copyPolygonsDtoH(tracts, colorMap) {
  console.time(`copy census tracts DtoH (${tracts.numRows.toLocaleString()} tracts)`);
  const result = [...tracts.toArrow()].map(({id, polygon}) => {
    return {
      id,
      color: colorMap.get(id),
      rings: [...polygon].map((ring) => [...ring].map(({x, y}) => [x, y])),
    };
  });
  console.timeEnd(`copy census tracts DtoH (${tracts.numRows.toLocaleString()} tracts)`);
  return result;
}

/**
 * @param {DataFrame<{ id: Int32, polygon: List<List<Struct<{x: Float32, y: Float32}>>> }>} tracts
 * @param {Map<number, [number, number, number, number]>} colorMap
 */
export function copyPolygonVerticesDtoH(tracts, colorMap) {
  console.time(`copy census tract vertices DtoH (${tracts.numRows.toLocaleString()} tracts)`);
  const result = [...tracts.toArrow()].flatMap(({id, polygon}) => {
    const color = colorMap.get(id);
    return [...polygon].flatMap((ring) => [...ring].map(({x, y}) => ({
                                                          id,
                                                          color,
                                                          position: [x, y, 0],
                                                          radius: Math.max(3, 2 + (id % 7)),
                                                          strokeWidth: Math.max(1, (id % 3)),
                                                        })));
  });
  console.timeEnd(`copy census tract vertices DtoH (${tracts.numRows.toLocaleString()} tracts)`);
  return result;
}

const sleep = (t) => new Promise((r) => setTimeout(r, t));

/**
 *
 * @param {DataFrame<{ x: Float32; y: Float32; }>} points
 * @param {Series<Uint32>} polyIds
 * @returns
 */
export function getPointsInBatches(points, colors, polyIds) {
  return {
    async * [Symbol.asyncIterator]() {
      const count  = Math.ceil(points.numRows / 1e6);
      const length = Math.floor(points.numRows / count);
      for (let offset = 0; offset < (count * length); offset += length) {
        yield {
          length,
          offset,
          attributes: {
            nodeFillColors: colors.data.subarray(offset, offset + length),
            nodeElementIndices: polyIds.data.subarray(offset, offset + length),
            nodeXPositions: points.get('x').data.subarray(offset, offset + length),
            nodeYPositions: points.get('y').data.subarray(offset, offset + length),
            nodeRadius: Series.sequence({type: new Uint8, init: 5, size: length, step: 0}).data,
          }
        };
        await sleep(16);
      }
    }
  };
}
