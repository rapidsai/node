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
    edgeListAccessor,
    edgeComponentAccessor,
    edgeSourceColorAccessor,
    edgeTargetColorAccessor,
} from './edges/attributes';

const NUM_SEGMENTS = 40;

export class EdgeLayer extends Layer {
    static getAccessors({ gl }) {
        return {
            instanceEdges: { ...edgeListAccessor(gl), accessor: 'getInstanceEdge' },
            instanceSourceColors: { ...edgeSourceColorAccessor(gl), accessor: 'getInstanceSourceColors' },
            instanceTargetColors: { ...edgeTargetColorAccessor(gl), accessor: 'getInstanceTargetColors' },
            instanceControlPoints: { ...edgeComponentAccessor(gl), accessor: 'getInstanceControlPoints' },
            instanceSourcePositions: { ...edgeComponentAccessor(gl), accessor: 'getInstanceSourcePositions' },
            instanceTargetPositions: { ...edgeComponentAccessor(gl), accessor: 'getInstanceTargetPositions' },
        };
    }
    initializeState(context) {
        this.internalState.selectedEdgeId = -1;
        this.internalState.highlightedEdgeId = -1;
        this.internalState.selectedEdgeIndex = -1;
        this.internalState.highlightedEdgeIndex = -1;
        this.internalState.selectedSourceNodeId = -1;
        this.internalState.highlightedSourceNodeId = -1;
        this.internalState.selectedTargetNodeId = -1;
        this.internalState.highlightedTargetNodeId = -1;
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
                highlightedNode: this.props.highlightedNode,
                strokeWidth: Math.max(1, this.props.width || 1),
                opacity: Math.max(0, Math.min(this.props.opacity, 1)),
            }
        });
    }
    getPickingInfo({ mode, info }) {
        if (info.index === -1) {
            info.edgeId = -1;
            info.sourceNodeId = -1;
            info.targetNodeId = -1;
        } else if (info.index === this.internalState.highlightedEdgeIndex) {
            info.edgeId = this.internalState.highlightedEdgeIndex;
            info.sourceNodeId = this.internalState.highlightedSourceNodeId;
            info.targetNodeId = this.internalState.highlightedTargetNodeId;
        } else {
            info.edgeId = info.index;
            const { buffer, offset = 0 } = this.props.data.attributes.instanceEdges;
            ([info.sourceNodeId, info.targetNodeId] = buffer.getData({
                length: 2, srcByteOffset: offset + (info.index * buffer.accessor.BYTES_PER_VERTEX),
            }));
        }
        this.internalState.highlightedEdgeId = info.edgeId;
        this.internalState.highlightedEdgeIndex = info.index;
        this.internalState.highlightedSourceNodeId = info.sourceNodeId;
        this.internalState.highlightedTargetNodeId = info.targetNodeId;
        if (mode === 'click') {
            this.internalState.selectedEdgeId = this.internalState.highlightedEdgeId;
            this.internalState.selectedEdgeIndex = this.internalState.highlightedEdgeIndex;
            this.internalState.selectedSourceNodeId = this.internalState.highlightedSourceNodeId;
            this.internalState.selectedTargetNodeId = this.internalState.highlightedTargetNodeId;
        }

        return info;
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
    highlightedNode: { type: 'number', min: -1, max: Number.MAX_SAFE_INTEGER, value: -1 },
};
