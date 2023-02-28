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

import './App.css';

import React, { useEffect, useState } from 'react';

import { readCsv, release, createQuadtree, setPolygon, getQuadtreePointCount, getQuadtreePoints } from './requests.js';

function Points({ props, buffer }) {
  const [csvLoaded, setCsvLoaded] = useState(false);
  const [csvName, setCsvName] = useState(null);

  /*
   load csv
   */
  useEffect(() => {
    // read CSV
    const preloadAllPoints = async () => {
      const csv = await readCsv('shuffled.csv');
      const quadtreeName = await createQuadtree(csv, { 'x': 'Longitude', 'y': 'Latitude' });
      const polygon = await setPolygon('test', [-127, 51, -64, 51, -64, 24, -127, 24, -127, 51]);
      const quadtreePointCount = await getQuadtreePointCount(quadtreeName, polygon);
      console.log('quadtreePointCount', quadtreePointCount);
      console.log(quadtreePointCount);
      const points = await getQuadtreePoints(quadtreeName, polygon, quadtreePointCount.count);
      console.log('points', points);
      console.log(buffer);
      props.pointOffset = quadtreePointCount.count;
      buffer.subdata(points, 0);
    }
    preloadAllPoints();
    return () => {
      release();
    }
  }, [buffer]);

  useEffect(() => {
    if (csvName) {
      console.log('csv loaded');
      setCsvLoaded(true);
    }
    else {
      console.log('csv not loaded');
    }
  }, [csvName]);

  return <div className="nodisplay" />
}

export default Points;
