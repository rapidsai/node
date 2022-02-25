// Copyright (c) 2022, NVIDIA CORPORATION.
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

import {DeckContext, DeckLayer, UpdateStateProps} from '../../deck.gl';
const {Layer, project32, gouraudLighting, UNIT} = require('@deck.gl/core');
import GL from '@luma.gl/constants';
import {Geometry, Model} from '@luma.gl/engine';

import {pointColorAccessor, pointNormalizeAccessor, pointPositionAccessor} from './attributes';

import fs from './point-cloud-layer-fragment.glsl';
import vs from './point-cloud-layer-vertex.glsl';

const DEFAULT_COLOR  = [0, 0, 0, 255];
const DEFAULT_NORMAL = [0, 0, 1];

function normalizeData(data: any) {
  const {header, attributes} = data;
  if (!header || !attributes) { return; }

  data.length = header.vertexCount;

  if (attributes.POSITION) { attributes.instancePositions = attributes.POSITION; }
  if (attributes.NORMAL) { attributes.instanceNormals = attributes.NORMAL; }
  if (attributes.COLOR_0) { attributes.instanceColors = attributes.COLOR_0; }
}

export class PointCloudGPUBase extends (Layer as typeof DeckLayer) {
  static get layerName() { return 'PointCloudGPUBase'; }
  getShaders() { return super.getShaders({vs, fs, modules: [project32, gouraudLighting]}); }
  static get defaultProps() {
    return {
      sizeUnits: 'pixels',
      opacity: {type: 'number', min: 0, max: 1, value: 0.5},
      pointSize: {type: 'number', min: 0, value: 10},  //  point radius in pixels
      material: true,
      getPositionX: {type: 'accessor', value: (x: any) => x.positionX},
      getPositionY: {type: 'accessor', value: (x: any) => x.positionY},
      getPositionZ: {type: 'accessor', value: (x: any) => x.positionZ},
      getColor: {type: 'accessor', value: DEFAULT_COLOR},
      getNormal: {type: 'accessor', value: DEFAULT_NORMAL},
      // Depreated
      radiusPixels: {deprecatedFor: 'pointSize'}
    };
  }

  static getAccessors({gl}: {gl: WebGL2RenderingContext}) {
    return {
      instancePositionsX: {...pointPositionAccessor(gl), accessor: 'getPositionX'},
      instancePositionsY: {...pointPositionAccessor(gl), accessor: 'getPositionY'},
      instancePositionsZ: {...pointPositionAccessor(gl), accessor: 'getPositionZ'},
      instanceColors: {...pointColorAccessor(gl), accessor: 'getColor'},
      instanceNormals: {...pointNormalizeAccessor(gl), accessor: 'getNormal'},
    };
  }

  initializeState(context: DeckContext) {
    /* eslint-disable max-len */
    this.getAttributeManager().addInstanced(PointCloudGPUBase.getAccessors(context));
    /* eslint-enable max-len */
  }

  updateState({props, oldProps, context, changeFlags}: UpdateStateProps) {
    super.updateState({props, oldProps, context, changeFlags});
    if (changeFlags.extensionsChanged) {
      this.state.model?.delete();
      this.state.model = this._getModel(context.gl);
      this.getAttributeManager().invalidateAll();
    }
    if (changeFlags.dataChanged) { normalizeData(props.data); }
  }

  draw({uniforms, ...rest}: {uniforms?: any, context?: DeckContext} = {}) {
    const {pointSize, sizeUnits} = this.props;
    this.state.model.draw({
      ...rest,
      uniforms: {
        sizeUnits: UNIT[sizeUnits],
        radiusPixels: pointSize,
        ...uniforms,
      }
    });
  }

  _getModel(gl: WebGL2RenderingContext) {
    // a triangle that minimally cover the unit circle
    const positions = [];
    for (let i = 0; i < 3; i++) {
      const angle = (i / 3) * Math.PI * 2;
      positions.push(Math.cos(angle) * 2, Math.sin(angle) * 2, 0);
    }

    return new Model(gl, {
      id: this.props.id,
      ...this.getShaders(),
      isInstanced: true,
      geometry: new Geometry(
        {drawMode: GL.TRIANGLES, attributes: {positions: new Float32Array(positions)}}),
    });
  }
}
