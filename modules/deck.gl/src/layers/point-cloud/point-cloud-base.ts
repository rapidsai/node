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
const {Layer, project32, gouraudLighting, picking, UNIT} = require('@deck.gl/core');
import {Geometry, Model} from '@luma.gl/engine';

import {pointColorAccessor, pointPositionAccessor} from './attributes';

import pointFragmentShader from './point-cloud-layer-fragment.glsl';
import pointVertexShader from './point-cloud-layer-vertex.glsl';

// const DEFAULT_COLOR  = [0, 0, 0, 255];
// const DEFAULT_NORMAL = [0, 0, 1];

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

  static get defaultProps() {
    return {
      sizeUnits: 'pixels',
      pointSize: {type: 'number', min: 0, value: 10},  //  point radius in pixels
      material: true,
      // Depreated
      radiusPixels: {deprecatedFor: 'pointSize'}
    };
  }

  static getAccessors({gl}: {gl: WebGLRenderingContext}) {
    return {
      instancePositionsX: {...pointPositionAccessor(gl), accessor: 'getPositionX'},
      instancePositionsY: {...pointPositionAccessor(gl), accessor: 'getPositionY'},
      instancePositionsZ: {...pointPositionAccessor(gl), accessor: 'getPositionZ'},
      instanceColors: {...pointColorAccessor(gl), accessor: 'getColor'},
      // instanceNormals:
      //   {size: 3, transition: true, accessor: 'getNormal', defaultValue: DEFAULT_NORMAL},
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
      this.state.model = this._getModel(context);
      this.getAttributeManager().invalidateAll();
    }
    if (changeFlags.dataChanged) { normalizeData(props.data); }
  }

  draw({uniforms, ...rest}: {uniforms?: any, context?: DeckContext} = {}) {
    const {pointSize, sizeUnits} = this.props;

    this.state.model.draw(
      {...rest, uniforms: {sizeUnits: UNIT[sizeUnits], radiusPixels: pointSize, ...uniforms}});
  }

  _getModel({gl, shaderCache}: DeckContext) {
    return new Model(gl, <any>{
      id: this.props.id,
      shaderCache,
      modules: [project32, picking, gouraudLighting],
      vs: pointVertexShader,
      fs: pointFragmentShader,
      // isIndexed: true,
      isInstanced: true,
      indexType: gl.UNSIGNED_INT,
      geometry: new Geometry({
        drawMode: gl.TRIANGLE_FAN,
        vertexCount: 4,
        attributes:
          {positions: {size: 3, value: new Float32Array([-1, -1, 0, -1, 1, 0, 1, 1, 0, 1, -1, 0])}}
      }),
    });
  }
}
