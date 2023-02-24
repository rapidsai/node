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

import React, { useEffect, useRef, createContext, useState } from 'react';
import regl from 'regl';
import { release, readCsv } from './requests';
import { fetchQuadtree } from './points';
const { getPointsViewMatrix, getPointsProjectionMatrix } = require('./matrices');

let testBuffer = [
  [-100, 37, -101, 37, -102, 37, -103, 37, -104, 37],
  [-100, 37, -100, 36, -100, 35, -100, 34, -100, 37],
  [-100, 37, -99, 37, -98, 37, -97, 37, -96, 37],
  [-100, 37, -100, 38, -100, 39, -100, 40, -100, 41],
];

const drawBufferObj = (buffer, props) => {
  return {
    vert: `
        precision mediump float;
        attribute vec2 pos;
        uniform float scale;
        uniform float time;
        uniform mat4 view, projection;
        varying vec3 fragColor;
        void main() {
          vec2 position = pos.xy;
          gl_PointSize = scale;
          gl_Position = projection * view * vec4(position, 1, 1);
          fragColor = vec3(0, 0, 0);
        }`,
    frag: `
        precision lowp float;
        varying vec3 fragColor;
        void main() {
          if (length(gl_PointCoord.xy - 0.5) > 0.5) {
            discard;
          }
          gl_FragColor = vec4(fragColor, 0.5);
        }`,
    attributes: {
      pos: { buffer: buffer, stride: 8, offset: 0 },
    },
    uniforms: {
      view: ({ tick }, props) => getPointsViewMatrix(props),
      scale:
        ({ tick }, props) => { return 20 * Math.max(0.5, Math.pow(props.zoomLevel, 1 / 2.6)); },
      projection: ({ viewportWidth, viewportHeight }) => getPointsProjectionMatrix(props),
      time: ({ tick }) => tick * 0.001
    },
    count: 5, //props.pointOffset,
    primitive: 'points'
  }
}

const ParticlesContext = createContext();

let useEffectNum = 0;

function Particles({ props }) {
  const canvasRef = useRef(null);
  const [reglState, setReglState] = useState({ reglInstance: null, buffer: null });

  useEffect(() => {
    // Create the initial regl instanc and the maximum size buffer for point storage.
    const reglInstance = regl({
      canvas: canvasRef.current,
    });
    const buffer = reglInstance.buffer({ usage: 'dynamic', type: 'float', length: 200000000 });
    setReglState({ reglInstance, buffer });
    return () => {
      reglInstance.destroy();
    }
  }, []);

  useEffect(() => {
    // initial rendering
    const { reglInstance, buffer } = reglState;
    useEffectNum += 1;
    if (buffer) {
      reglState.buffer.subdata(testBuffer[useEffectNum % 4], 0);
      const drawBuffer = reglInstance(drawBufferObj(buffer, props));
      drawBuffer(props);

      const readPoints = async (inputCsv) => {
        // load the csv
        // set the polygon
        // read the points
      }
      readPoints('shuffled.csv');
    }
  }, [props]);

  return <ParticlesContext.Provider value={{ reglState, setReglState }}>
    <canvas ref={canvasRef} className='foreground-canvas' width="900" height="900" />;
  </ParticlesContext.Provider>
}

export default Particles;
