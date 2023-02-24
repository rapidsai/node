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

/*
 * Return the points contained within a specific quadrant for display
 * @param {number} quadrant - the quadrant to return points for
 * @returns {ArrowBuffer} - an array of points
 * @example
 * const points = points(1);
 * console.log(points);
 * // => [[-100, 37], [-101, 37], [-102, 37], [-103, 37], [-104, 37]]
 */
import {
  createQuadtree,
  getQuadtreePointCount,
  getQuadtreePoints,
  readCsv,
  release,
  setPolygon
} from './requests';

const quadPair = [[-127, 25, -63, 49]];

const makeQuadrants = (rectangle, list, depth = 3) => {
  const x1 = rectangle[0];
  const y1 = rectangle[1];
  const x2 = rectangle[2];
  const y2 = rectangle[3];
  const mx = (x1 + x2) / 2;
  const my = (y1 + y2) / 2;
  if (depth < 1) {
    list.unshift([x1, y1, x2, y1, x2, y2, x1, y2, x1, y1]);
    return;
  }
  makeQuadrants([x1, y1, mx, my], list, depth - 1);
  makeQuadrants([mx, y1, x2, my], list, depth - 1);
  makeQuadrants([x1, my, mx, y2], list, depth - 1);
  makeQuadrants([mx, my, x2, y2], list, depth - 1);
};
function shuffleArray(array) {
  for (let i = array.length - 1; i > 0; i--) {
    const j              = Math.floor(Math.random() * (i + 1));
    [array[i], array[j]] = [array[j], array[i]];
  }
};
async function getPolygonSizes(quadtreeName, polygons, props) {
  const sizes = {};
  for (let i = 0; i < polygons.length; i++) {
    const polygonName  = await setPolygon(polygons[i], polygons[i]);
    const res          = await getQuadtreePointCount(quadtreeName, polygonName);
    const size         = res.count;
    quads[polygons[i]] = {totalPoints: size, pointOffsets: [], loadedPoints: 0};
  }
};

const quads = {};
const done  = {};

const quadIsFull = (quad) => {
  console.log(quads[quad]);
  // if the polygon has been fully loaded, remove it from the list
  if (quads[quad].totalPoints <= quads[quad].loadedPoints) {
    done[quad] = quads[quad];
    delete quads[quad];
  }
};

export const fetchQuadtree = async (csvName, engine, props) => {
  // API to create quadtree from CSV file
  const quadtreeName = await createQuadtree(csvName, {x: 'Longitude', y: 'Latitude'});

  // Subdivide the intial viewport quadPair into many quadrants of depth 3
  // and shuffle it
  quads[quadPair] = {};
  let polygons    = quads;

  await getPolygonSizes(quadtreeName, polygons, props);

  // Iterate over the polygons and fetch points from the server
  let which = 0;
  while (props.pointOffset < props.pointBudget && polygons.length > 0) {
    if (quadIsFull(polygons[which])) {
      polygons.splice(which, 1);
      which = which % polygons.length;
      continue;
    }
    const hostPoints = await points(quadtreeName, polygons[which], props);
    await updateQuad(polygons[which], hostPoints, props);
  }
};

const points = async (quadtreeName, quadrant, props) => {
  const hostPoints = await getQuadtreePoints(quadtreeName, quadrant, props.pointsPerRequest);
  return hostPoints;
};

const updateQuad = async (polygon, points, props) => {
  // Update the point offset and loaded points
  const newOffset = (props.pointOffset % props.pointBudget) + points.length;
  // quad is not done
  if (quads[polygon]) {
    if (quads[polygon].loadedPoints === undefined) { quads[polygon].loadedPoints = 0; }
    quads[polygon].loadedPoints += points.length / 2
    quads[polygon].pointOffsets.push([props.pointOffset, newOffset]);
  }
  props.pointOffset = newOffset;
};
