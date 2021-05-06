// Copyright (c) 2015 - 2017 Uber Technologies, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

// Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

import { Layer, project32, picking } from '@deck.gl/core';
import GL from '@luma.gl/constants';
import { Model, Geometry } from '@luma.gl/core';

import vs from './node-layer-vertex.glsl';
import fs from './node-layer-fragment.glsl';

const DEFAULT_COLOR = [0, 0, 0, 255];

const defaultProps = {
  radiusScale: { type: 'number', min: 0, value: 1 },
  radiusMinPixels: { type: 'number', min: 0, value: 0 }, //  min point radius in pixels
  radiusMaxPixels: { type: 'number', min: 0, value: Number.MAX_SAFE_INTEGER }, // max point radius in pixels

  lineWidthUnits: 'meters',
  lineWidthScale: { type: 'number', min: 0, value: 1 },
  lineWidthMinPixels: { type: 'number', min: 0, value: 0 },
  lineWidthMaxPixels: { type: 'number', min: 0, value: Number.MAX_SAFE_INTEGER },

  stroked: false,
  filled: true,

  getPosition: { type: 'accessor', value: x => x.position },
  getRadius: { type: 'accessor', value: 1 },
  getFillColor: { type: 'accessor', value: DEFAULT_COLOR },
  getLineColor: { type: 'accessor', value: DEFAULT_COLOR },
  getLineWidth: { type: 'accessor', value: 1 },

  // deprecated
  strokeWidth: { deprecatedFor: 'getLineWidth' },
  outline: { deprecatedFor: 'stroked' },
  getColor: { deprecatedFor: ['getFillColor', 'getLineColor'] }
};

export default class NodeLayer extends Layer {
  getShaders(id) {
    return super.getShaders({ vs, fs, modules: [project32, picking] });
  }

  initializeState() {
    this.getAttributeManager().addInstanced({
      instancePositions: {
        size: 2,
        type: GL.FLOAT,
        fp64: this.use64bitPositions(),
        transition: false,
        accessor: 'getPosition'
      },
      instanceRadius: {
        size: 1,
        transition: false,
        accessor: 'getRadius',
        type: GL.UNSIGNED_BYTE,
        defaultValue: 1
      },
      instanceFillColors: {
        size: this.props.colorFormat.length,
        transition: false,
        normalized: true,
        type: GL.UNSIGNED_BYTE,
        accessor: 'getFillColor',
        defaultValue: [0, 0, 0, 255]
      },
      instanceLineColors: {
        size: this.props.colorFormat.length,
        transition: false,
        normalized: true,
        type: GL.UNSIGNED_BYTE,
        accessor: 'getLineColor',
        defaultValue: [0, 0, 0, 255]
      },
      instanceLineWidths: {
        size: 1,
        transition: false,
        accessor: 'getLineWidth',
        defaultValue: 1
      }
    });
  }

  updateState({ props, oldProps, changeFlags }) {
    super.updateState({ props, oldProps, changeFlags });
    if (changeFlags.extensionsChanged) {
      const { gl } = this.context;
      if (this.state.model) {
        this.state.model.delete();
      }
      this.setState({ model: this._getModel(gl) });
      this.getAttributeManager().invalidateAll();
    }
  }

  draw({ uniforms }) {
    const { viewport } = this.context;
    const {
      radiusScale,
      radiusMinPixels,
      radiusMaxPixels,
      stroked,
      filled,
      lineWidthUnits,
      lineWidthScale,
      lineWidthMinPixels,
      lineWidthMaxPixels
    } = this.props;

    const widthMultiplier = lineWidthUnits === 'pixels' ? viewport.metersPerPixel : 1;

    this.state.model
      .setUniforms(uniforms)
      .setUniforms({
        stroked: stroked ? 1 : 0,
        filled,
        radiusScale,
        radiusMinPixels,
        radiusMaxPixels,
        lineWidthScale: lineWidthScale * widthMultiplier,
        lineWidthMinPixels,
        lineWidthMaxPixels
      })
      .draw();
  }

  _getModel(gl) {
    // a square that minimally cover the unit circle
    const positions = [-1, -1, 0, -1, 1, 0, 1, 1, 0, 1, -1, 0];

    return new Model(
      gl,
      Object.assign(this.getShaders(), {
        id: this.props.id,
        geometry: new Geometry({
          drawMode: GL.TRIANGLE_FAN,
          vertexCount: 4,
          attributes: {
            positions: { size: 3, value: new Float32Array(positions) }
          }
        }),
        isInstanced: true
      })
    );
  }
}

NodeLayer.layerName = 'NodeLayer';
NodeLayer.defaultProps = defaultProps;
