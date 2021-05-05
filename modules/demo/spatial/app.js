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

import React from 'react';

import { readFileSync } from 'fs';
import { Table } from 'apache-arrow';

import DeckGL from '@deck.gl/react';
import { OrthographicView } from '@deck.gl/core';
import { PolygonLayer, ScatterplotLayer } from '@deck.gl/layers';

import { DataFrame, Series, Float32, Uint32 } from '@rapidsai/cudf';
import { Quadtree } from '@rapidsai/cuspatial';

import { ColorMapper } from './color';

const colorMap = new ColorMapper();

const { tracts } = (() => {
  const table = Table.from(readFileSync('data/263_tracts.arrow'));
  console.time(`load geometry (${table.length.toLocaleString()} polys)`);
  const tracts = new DataFrame({
    id: Series.new(table.getChildAt(0)),
    polygon: Series.new(table.getChildAt(1)),
  });
  console.timeEnd(`load geometry (${table.length.toLocaleString()} polys)`);
  return { tracts };
})();

const { polyIds, points } = (() => {

  const points = ((table) => {

    console.time(`load points (${table.length.toLocaleString()} points)`);

    const points = new DataFrame({
      x: Series.new(table.slice(0, table.length / 10 | 0).getChildAt(0)),
      y: Series.new(table.slice(0, table.length / 10 | 0).getChildAt(1)),
    });

    console.timeEnd(`load points (${table.length.toLocaleString()} points)`);

    return points;
  })(Table.from(readFileSync('data/168898952_points.arrow')));

  const [xMin, xMax] = points.get('x').minmax();
  const [yMin, yMax] = points.get('y').minmax();

  console.time(`construct quadtree (${points.numRows.toLocaleString()} points)`);

  const quadtree = Quadtree.new({
    x: points.get('x'),
    y: points.get('y'),
    xMin, xMax, yMin, yMax,
    scale: 1, maxDepth: 15, minSize: 512,
  });

  console.timeEnd(`construct quadtree (${points.numRows.toLocaleString()} points)`);

  console.time(`filter points (${points.numRows.toLocaleString()} points)`);

  // Color points by which census tract they're in
  const polyAndPointIdxs = quadtree.pointInPolygon(tracts.get('polygon'));
  const polyIds = polyAndPointIdxs.get('polygon_index');

  // Color points by the census tract they're closest too
  // const polyAndPointIdxs = quadtree.pointToNearestPolyline(tracts.get('polygon').elements, 1);
  // const polyIds = polyAndPointIdxs.get('polyline_index');

  const filteredPoints = quadtree.points.gather(polyAndPointIdxs.get('point_index'));

  console.timeEnd(`filter points (${points.numRows.toLocaleString()} points)`);

  return { polyIds, points: filteredPoints };
})();

const [xMin, xMax] = tracts.get('polygon').elements.elements.getChild('x').minmax();
const [yMin, yMax] = tracts.get('polygon').elements.elements.getChild('y').minmax();

export default class App extends React.Component {
  render() {

    const [width, height] = [xMax - xMin, yMax - yMin];
    const INITIAL_VIEW_STATE = {
      minZoom: Number.NEGATIVE_INFINITY,
      maxZoom: Number.POSITIVE_INFINITY,
      target: [
        xMin + width * 0.5,
        yMin + height * 0.5,
        0
      ],
      zoom: (() => {
        const { outerWidth, outerHeight } = window;
        const world = (width > height ? width : height);
        const screen = (width > height ? outerWidth : outerHeight);
        const zoom = (world > screen ? -(world / screen) : (screen / world)) * 2;
        return Math.log2(Math.abs(zoom)) * Math.sign(zoom);
      })(),
    };

    return (
      <DeckGL
        controller={{ keyboard: false }}
        initialViewState={INITIAL_VIEW_STATE}
        views={[
          new OrthographicView({
            clear: {
              color: [...[46, 46, 46].map((x) => x / 255), 1]
            }
          })
        ]}
        layers={
          [
            // Census tracts
            new PolygonLayer({
              filled: true,
              stroked: true,
              extruded: false,
              positionFormat: `XY`,
              lineWidthMinPixels: 1,
              data: copyPolygonsDtoH(tracts),
              getPolygon: ({ rings }) => rings,
              getElevation: ({ id }) => id,
              getFillColor: ({ color }) => [...color.slice(0, 3), 15],
              getLineColor: ({ color }) => [...color.slice(0, 3), 255],
            }),
            // Polygon vertices
            new ScatterplotLayer({
              id: 'polygon_vertices',
              filled: false,
              stroked: true,
              radiusMinPixels: 3,
              getRadius: (x) => x.radius,
              getLineColor: (x) => x.color,
              getPosition: (x) => x.position,
              getLineWidth: (x) => x.strokeWidth,
              data: copyPolygonVerticesDtoH(tracts),
            }),
            // Taxi rides
            new ScatterplotLayer({
              id: 'taxi_rides',
              filled: true,
              stroked: false,
              _normalize: false,
              radiusMinPixels: 0,
              radiusUnits: 'pixels',
              data: copyPointsDtoH(polyIds, points),
            }),
          ]}
      />
    );
  }
}

function copyPolygonsDtoH(tracts) {
  return tracts.toArrow().toArray().map(({ id, polygon }) => {
    return {
      id,
      color: colorMap.get(id),
      rings: polygon.toArray().map((ring) => {
        return ring.toArray().map(({ x, y }) => [x, y]);
      }),
    };
  });
}

function copyPolygonVerticesDtoH(tracts) {
  return tracts.toArrow().toArray().flatMap(({ id, polygon }) => {
    const color = colorMap.get(id);
    return polygon.toArray().flatMap((ring) => {
      return ring.toArray().map(({ x, y }) => ({
        id,
        color,
        position: [x, y, 0],
        radius: Math.max(3, 2 + (id % 7)),
        strokeWidth: Math.max(1, (id % 3)),
      }))
    });
  });
}

function copyPointsDtoH(polyIds, points) {

  console.time(`compute positions + colors (${points.numRows.toLocaleString()} points)`);

  const positions = (() => {
    const size = points.numRows;

    let positions = Series.sequence({ type: new Float32, init: 0, size: size * 3, step: 0 });
    positions = positions.scatter(points.get('x'), Series.sequence({ type: new Uint32, init: 0, size, step: 3 }));
    positions = positions.scatter(points.get('y'), Series.sequence({ type: new Uint32, init: 1, size, step: 3 }));
    return positions;
  })();

  const colors = (() => {
    let palette = Array.from({ length: polyIds.nunique() }, (_, i) => colorMap.get(i));
    palette = new Uint32Array(new Uint8Array([].concat.apply([], palette)).buffer);
    palette = Series.new({ type: new Uint32, data: palette });
    return palette.gather(polyIds);
  })();

  console.timeEnd(`compute positions + colors (${points.numRows.toLocaleString()} points)`);

  return {
    length: points.numRows,
    attributes: {
      instancePositions: {
        size: 3,
        value: positions.data.toArray(),
      },
      instanceFillColors: {
        size: 4,
        normalized: true,
        value: new Uint8Array(colors.data.toArray().buffer),
      },
    }
  };
}
