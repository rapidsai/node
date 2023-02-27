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
import mat4 from 'gl-mat4';
const { getProjection, getPointsViewMatrix, getPointsProjectionMatrix } = require('./matrices');

let testBuffer = [
  [-100, 37, -101, 37, -102, 37, -103, 37, -104, 37],
  [-100, 37, -100, 36, -100, 35, -100, 34, -100, 37],
  [-100, 37, -99, 37, -98, 37, -97, 37, -96, 37],
  [-100, 37, -100, 38, -100, 39, -100, 40, -100, 41],
];

const drawBufferObj = (buffer, props) => {
  //const world = [props.map.transform._center.lng, props.map.transform._center.lat, 0, 1];
  /*
  const world = [-105, 40, 0, 1];
  const mercator = [
    (180.0 + world[0]) / 360.0,
    (180.0 - (180.0 / Math.PI * Math.log(Math.tan(Math.PI * 0.25 + world[1] * (Math.PI / 360.0))))) / 360.0, 0, 1
  ];
  const mercatorCoord = mat4.multiply([], props.map.transform.mercatorMatrix, mercator);
  console.log(props.map.project([world[0], world[1]]));
  const labelPlane = mat4.multiply([], props.map.transform.labelPlaneMatrix, mercatorCoord);
  console.log([labelPlane[0] / labelPlane[3], labelPlane[1] / labelPlane[3]]);
  console.log(labelPlane);
  console.log(props.map.transform.labelPlaneMatrix);
  */
  console.log(props.map.transform.width);
  return {
    vert: `
        precision mediump float;
        attribute vec2 pos;
        uniform float scale;
        uniform float time;
        uniform mat4 view, projection, screenToClip;
        varying vec3 fragColor;
        # define PI 3.1415926535897932384626433832795
        void main() {
          vec2 position = pos.xy;
          gl_PointSize = scale;
          position.x = (180.0 + position.x) / 360.0;
          position.y = (180.0 - (180.0 / PI * log(tan(PI * 0.25 + position.y * (PI / 360.0))))) / 360.0;
          vec4 screen = view * projection * vec4(position, 0, 1);
          gl_Position = screenToClip * screen;
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
      screenToClip: () => getProjection(
        [-1, -1, 0, 1, -1, 1, 0, 1, 1, -1, 0, 1, 1, 1, 0, 1],
        [
          0, props.map.transform.height, 0, 1,
          0, 0, 0, 1,
          props.map.transform.width, props.map.transform.height, 0, 1,
          props.map.transform.width, 0, 0, 1
        ],
      ),
      view: ({ tick }, props) => props.map.transform.labelPlaneMatrix,
      scale:
        ({ tick }, props) => { return 20 * Math.max(0.5, Math.pow(props.zoomLevel, 1 / 2.6)); },
      projection: ({ viewportWidth, viewportHeight }) => props.map.transform.mercatorMatrix,
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
