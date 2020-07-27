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
import edgeVertexShader from './edges/edge-vertex.glsl';
import edgeFragmentShader from './edges/edge-fragment.glsl';
import {
    edgeComponentAccessor,
    edgeSourceColorAccessor,
    edgeTargetColorAccessor,
} from './edges/attributes';

const NUM_SEGMENTS = 40;

export class EdgeLayer extends Layer {
    static getAccessors({ gl }) {
        return {
            instanceSourceColors: { ...edgeSourceColorAccessor(gl), accessor: 'getInstanceSourceColors' },
            instanceTargetColors: { ...edgeTargetColorAccessor(gl), accessor: 'getInstanceTargetColors' },
            instanceControlPoints: { ...edgeComponentAccessor(gl), accessor: 'getInstanceControlPoints' },
            instanceSourcePositions: { ...edgeComponentAccessor(gl), accessor: 'getInstanceSourcePositions' },
            instanceTargetPositions: { ...edgeComponentAccessor(gl), accessor: 'getInstanceTargetPositions' },
        };
    }
    initializeState(context) {
        this.getAttributeManager().addInstanced(EdgeLayer.getAccessors(context));
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
                ...uniforms,
                strokeWidth: Math.max(1, this.props.width || 1),
                opacity: Math.max(0, Math.min(this.props.opacity, 1)),
            }
        });
    }
    _getModel({ gl, shaderCache }) {
        return new Model(gl, {
            id: this.props.id,
            shaderCache,
            modules: [project32, picking],
            vs: edgeVertexShader,
            fs: edgeFragmentShader,
            uniforms: { numSegments: NUM_SEGMENTS },
            isInstanced: true,
            geometry: new Geometry({
                drawMode: gl.TRIANGLE_STRIP,
                attributes: {
                    //
                    // (0, -1)------------_(1, -1)
                    //      |         _,-'  |
                    //      o     _,-'      o
                    //      | _,-'          |
                    // (0,  1)-------------(1,  1)
                    //
                    positions: new Float32Array(
                        Array.from({ length: NUM_SEGMENTS })
                             .flatMap((_, i) => [i, -1, 0, i, 1, 0]))
                },
            }),
        });
    }
}

EdgeLayer.layerName = 'EdgeLayer';
EdgeLayer.defaultProps = {
    opacity: { type: 'number', min: 0, max: 1, value: 1 },
    width: { type: 'number', min: Number.MIN_VALUE, max: 100, value: 1 },
};
