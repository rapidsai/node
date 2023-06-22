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
import { State, ParticleState } from './types';
import { readCsv, release, createQuadtree, setPolygon, getQuadtreePointCount, getQuadtreePoints, setRapidsViewerDataframe, setRapidsViewerViewport, changeRapidsViewerBudget, getRapidsViewerNextPoints } from './requests.js';
import { mapPropsStream, createEventHandler } from 'recompose';
import * as ix from './ix';
import { ReglRingBuffer } from './ringbuffer';
const { getProjection } = require('./matrices');

let testBuffer = new Float32Array([
  [-100, 37, -101, 37, -102, 37, -103, 37, -104, 37],
  [-100, 37, -100, 36, -100, 35, -100, 34, -100, 37],
  [-100, 37, -99, 37, -98, 37, -97, 37, -96, 37],
  [-100, 37, -100, 38, -100, 39, -100, 40, -100, 41],
].flatMap((x) => x));

const drawBufferObj = (buffer: regl.Buffer, props: State) => {
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
      pos: { buffer: buffer as regl.Buffer, stride: 8, offset: 0 },
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
        ({ tick }, props) => { return Math.max(0.5, Math.pow(props.zoomLevel, 1 / 5)); },
      projection: ({ viewportWidth, viewportHeight }) => props.map.transform.mercatorMatrix,
      time: ({ tick }) => tick * 0.001
    },
    count: props.pointBudget,
    primitive: 'points'
  }
};

interface ParticlesContextType {
  reglState: { reglInstance: regl.Regl | null; buffer: ReglRingBuffer | null };
  setReglState: React.Dispatch<React.SetStateAction<{ reglInstance: regl.Regl | null; buffer: ReglRingBuffer | null }>>;
};
interface MapBounds {
  _sw: { lng: number, lat: number },
  _ne: { lng: number, lat: number }
}

const ParticlesContext = createContext<ParticlesContextType>({
  reglState: { reglInstance: null, buffer: null },
  setReglState: () => null
});
// mapPropsStream takes an observable (props$)
const withParticlesProps = mapPropsStream((props$) => {
  // Convert the Observable into an AsyncIterable
  const props_ = ix.ai.from<ParticleState>(props$ as AsyncIterable<ParticleState>);
  // Create a pipeline
  const sourceNameChanged = props_.pipe(
    // Don't execute the following steps in the pipeline unless the below condition occurs
    ix.ai.ops.distinctUntilChanged({
      comparer(x, y) {
        return x.sourceName === y.sourceName
      }
    }),
    // Execute the subsequent events and emit the next event in order
    ix.ai.ops.flatMap((props) => {
      // Create a function that emits AsyncIterables
      const computeInitialQuadtree = ix.ai.from(async function* () {
        // Execute the async first points load
        const csv = await readCsv(props.sourceName, ["Longitude", "Latitude"]);
        // Set the RapidsViewer dataframe
        await setRapidsViewerDataframe(csv, "Longitude", "Latitude");
        // Set the current viewport
        const mapBounds = props.map.getBounds() as MapBounds;
        const serverBounds = { lb: [mapBounds._sw.lng, mapBounds._sw.lat], ub: [mapBounds._ne.lng, mapBounds._ne.lat] };
        await setRapidsViewerViewport(serverBounds);
        // Set the budget
        await changeRapidsViewerBudget(props.pointBudget);
        const initial = new Float32Array(0);
        yield { initial, props };
      }()).pipe(
        // the next event in the chain will an async function that takes
        // quadtreeName, polygon, count, and props: an event emitted by the last step in the pipeline
        ix.ai.ops.switchMap(async ({ initial, props }) => {
          return {
            // return the original props, with the addition of the buffer created from getQuadtreePoints
            ...props, buffer: (
              await getRapidsViewerNextPoints(props.pointsPerRequest)
            ) as Float32Array
          } as ParticleState
        }),
      )
      // create an AsyncIterable of a ParticleState object and the events emitted by a
      return ix.ai.concat(
        ix.ai.of({ ...props, buffer: testBuffer } as ParticleState), computeInitialQuadtree
      )
    }),
  );

  // Watch for transform change
  const transformChanged = props_.pipe(
    ix.ai.ops.distinctUntilChanged({
      comparer(x, y) {
        return x.map.transform === y.map.transform
      }
    }),
    ix.ai.ops.flatMap((props) => {
      const nextPoints = ix.ai.from(async function* () {
        console.log('transform changed');
        const mapBounds = props.map.getBounds() as MapBounds;
        await setRapidsViewerViewport(mapBounds);
        debugger;
        yield { props };
      }()).pipe(
        ix.ai.ops.switchMap(async ({ props }) => {
          return {
            ...props,
          } as ParticleState
        }),
      )
      return ix.ai.concat(
        ix.ai.of({ ...props, buffer: testBuffer } as ParticleState), nextPoints
      )
    }),
  );

  // Watch for buffer input
  const bufferChanged = props_.pipe(
    ix.ai.ops.flatMap((props) => {
      const nextPoints = ix.ai.from(async function* () {
        const next = await getRapidsViewerNextPoints(props.pointsPerRequest);
        console.log('Point length: ', next.length);
        yield { next, props };
      }()).pipe(
        ix.ai.ops.switchMap(async ({ next, props }) => {
          return {
            ...props, buffer: (
              next
            ) as Float32Array
          } as ParticleState
        }),
      )
      return ix.ai.concat(
        ix.ai.of({ ...props, buffer: testBuffer } as ParticleState), nextPoints
      )
    }),
  );
  // Create a function that emits AsyncIterables
  // combine the AsyncIterables emitted from props_ and the AsyncIterables emitted from sourceNameChanged
  return ix.ai.combineLatest(
    props_,
    sourceNameChanged,
    transformChanged,
    bufferChanged
  ).pipe(
    // when an event is emitted, emit a new event with the destructured props and the buffer
    ix.ai.ops.map(([props, particleState]) => {
      return { ...props, buffer: particleState.buffer };
    })
    // Make this pipeline observable
  ).pipe(ix.ai.toObservable);
});

interface ReglState {
  reglInstance: regl.Regl | null;
  buffer: ReglRingBuffer | null;
}

let count = 0;

const Particles = withParticlesProps(
  function Particles({ loading, updatePointOffset, ...props }: ParticleState) {
    const canvasRef = useRef(null);
    const [reglState, setReglState] = useState<ReglState>({ reglInstance: null, buffer: null });
    let { reglInstance, buffer } = reglState;

    useEffect(() => {
      // Create the initial regl instanc and the maximum size buffer for point storage.
      console.log('Empty particles useEffect');
      reglInstance = regl({
        canvas: canvasRef.current as any,
      });
      buffer = new ReglRingBuffer(reglInstance, props.pointBudget);
      setReglState({ reglInstance, buffer });
      return () => {
        reglInstance!.destroy();
        buffer!.destroy();
      }
    }, []); // eslint-disable-line react-hooks/exhaustive-deps

    useEffect(() => {
      // initial rendering
      if (buffer && reglInstance) {
        //if (count < 5) {
        console.log(props.buffer.length, props.buffer[0]);
        buffer.write(props.buffer);
        //}
        //count++;
        props.pointOffset = props.buffer.length;
        const drawBuffer = reglInstance(drawBufferObj(buffer.get(), props) as regl.InitializationOptions);
        drawBuffer(props);
      }
    }, [props]); // eslint-disable-line react-hooks/exhaustive-deps

    return <ParticlesContext.Provider value={{ reglState, setReglState }}>
      <canvas ref={canvasRef} className='foreground-canvas' width="2000" height="2000" />
    </ParticlesContext.Provider>
  });

export default Particles;
