// Copyright (c) 2020, NVIDIA CORPORATION.
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

import { Model, Geometry } from '@luma.gl/engine';
import { Layer, picking, project32 } from '@deck.gl/core';
import nodeVertexShader from './nodes/node-vertex.glsl';
import nodeFragmentShader from './nodes/node-fragment.glsl';
import {
    nodeColorAccessor,
    nodeRadiusAccessor,
    nodePositionAccessor,
    // nodeElementIndicesAccessor,
} from './nodes/attributes';

export class NodeLayer extends Layer {
    static getAccessors({ gl }) {
        return {
            instanceRadius: { ...nodeRadiusAccessor(gl), accessor: 'getRadius' },
            instanceFillColors: { ...nodeColorAccessor(gl), accessor: 'getFillColor' },
            instanceLineColors: { ...nodeColorAccessor(gl), accessor: 'getLineColor' },
            instanceXPositions: { ...nodePositionAccessor(gl), accessor: 'getXPosition' },
            instanceYPositions: { ...nodePositionAccessor(gl), accessor: 'getYPosition' },
            // elementIndices: { ...nodeElementIndicesAccessor(gl), accessor: 'getElementIndex' },
        };
    }
    initializeState(context) {
        this.getAttributeManager().addInstanced(NodeLayer.getAccessors(context));
    }
    updateState({ props, oldProps, context, changeFlags }) {
        super.updateState({ props, oldProps, context, changeFlags });
        if (changeFlags.extensionsChanged) {
            if (this.state.model) {
                this.state.model.delete();
            }
            this.setState({ model: this._getModel(context) });
            this.getAttributeManager().invalidateAll();
        }
    }
    draw({ uniforms, ...rest } = {}) {
        this.state.model.draw({
            ...rest,
            uniforms: {
                filled: !!this.props.filled,
                stroked: +(!!this.props.stroked),
                fillOpacity: this.props.fillOpacity,
                strokeRatio: this.props.strokeRatio,
                strokeOpacity: this.props.strokeOpacity,
                radiusScale: this.props.radiusScale,
                lineWidthScale: this.props.lineWidthScale,
                radiusMinPixels: this.props.radiusMinPixels,
                radiusMaxPixels: this.props.radiusMaxPixels,
                lineWidthMinPixels: this.props.lineWidthMinPixels,
                lineWidthMaxPixels: this.props.lineWidthMaxPixels,
                ...uniforms,
            }
        });
    }
    _getModel({ gl, shaderCache }) {
        return new Model(gl, {
            id: this.props.id,
            shaderCache,
            modules: [project32, picking],
            vs: nodeVertexShader,
            fs: nodeFragmentShader,
            // isIndexed: true,
            isInstanced: true,
            indexType: gl.UNSIGNED_INT,
            geometry: new Geometry({
                drawMode: gl.TRIANGLE_FAN,
                vertexCount: 4,
                attributes: {
                    positions: {
                        size: 3,
                        value: new Float32Array([
                            -1, -1, 0,
                            -1,  1, 0,
                             1,  1, 0,
                             1, -1, 0
                        ])
                    }
                }
            }),
        });
    }
}

NodeLayer.layerName = 'NodeLayer';
NodeLayer.defaultProps = {
    filled: { type: 'boolean', value: true },
    stroked: { type: 'boolean', value: true },
    strokeRatio: { type: 'number', min: 0, max: 1, value: 0.05 },
    fillOpacity: { type: 'number', min: 0, max: 1, value: 1 },
    strokeOpacity: { type: 'number', min: 0, max: 1, value: 1 },
    radiusScale: { type: 'number', min: 0, value: 1 },
    lineWidthScale: { type: 'number', min: 0, value: 1 },
    radiusMinPixels: { type: 'number', min: 0, value: 0 }, //  min point radius in pixels
    radiusMaxPixels: { type: 'number', min: 0, value: Number.MAX_SAFE_INTEGER }, // max point radius in pixels
    lineWidthMinPixels: { type: 'number', min: 0, value: 0 },
    lineWidthMaxPixels: { type: 'number', min: 0, value: Number.MAX_SAFE_INTEGER },
  };
