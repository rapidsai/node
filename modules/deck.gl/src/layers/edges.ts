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

import {Geometry, Model} from '@luma.gl/engine';

import {DeckContext, DeckLayer, PickingInfo, UpdateStateProps} from '../deck.gl';

import {
  edgeComponentAccessor,
  edgeListAccessor,
  edgeSourceColorAccessor,
  edgeTargetColorAccessor,
} from './edges/attributes';
import edgeFragmentShader from './edges/edge-fragment.glsl';
import edgeVertexShader from './edges/edge-vertex.glsl';

const {Layer, picking, project32} = require('@deck.gl/core');

const NUM_SEGMENTS = 40;

export class EdgeLayer extends (Layer as typeof DeckLayer) {
  static get layerName() { return 'EdgeLayer'; }
  static get defaultProps() {
    return {
      width: 1,
      opacity: 1,
      highlightedNode: -1,
      highlightedEdge: -1,
    };
  }
  static getAccessors({gl}: {gl: WebGLRenderingContext}) {
    return {
      instanceEdges: {...edgeListAccessor(gl), accessor: 'getInstanceEdge'},
      instanceSourceColors: {...edgeSourceColorAccessor(gl), accessor: 'getInstanceSourceColors'},
      instanceTargetColors: {...edgeTargetColorAccessor(gl), accessor: 'getInstanceTargetColors'},
      instanceControlPoints: {...edgeComponentAccessor(gl), accessor: 'getInstanceControlPoints'},
      instanceSourcePositions:
        {...edgeComponentAccessor(gl), accessor: 'getInstanceSourcePositions'},
      instanceTargetPositions:
        {...edgeComponentAccessor(gl), accessor: 'getInstanceTargetPositions'},
    };
  }
  initializeState(context: DeckContext) {
    this.internalState.selectedEdgeId          = -1;
    this.internalState.highlightedEdgeId       = -1;
    this.internalState.selectedEdgeIndex       = -1;
    this.internalState.highlightedEdgeIndex    = -1;
    this.internalState.selectedSourceNodeId    = -1;
    this.internalState.highlightedSourceNodeId = -1;
    this.internalState.selectedTargetNodeId    = -1;
    this.internalState.highlightedTargetNodeId = -1;
    this.getAttributeManager().addInstanced(EdgeLayer.getAccessors(context));
  }
  updateState({props, oldProps, context, changeFlags}: UpdateStateProps) {
    ['selectedEdgeId',
     'highlightedEdgeId',
     'selectedEdgeIndex',
     'highlightedEdgeIndex',
     'selectedSourceNodeId',
     'highlightedSourceNodeId',
     'selectedTargetNodeId',
     'highlightedTargetNodeId']
      .filter((key) => typeof props[key] === 'number')
      .forEach((key) => this.internalState[key] = props[key]);

    // if (this.internalState.highlightedEdgeIndex && this.internalState.highlightedEdgeIndex !==
    // -1) {
    //   props.highlightedObjectIndex = this.internalState.highlightedEdgeIndex;
    // }

    super.updateState({props, oldProps, context, changeFlags});

    if (changeFlags.extensionsChanged) {
      if (this.state.model) { this.state.model.delete(); }
      this.setState({model: this._getModel(context)});
      this.getAttributeManager().invalidateAll();
    }
  }
  serialize() {
    return {
      selectedEdgeId: this.internalState.selectedEdgeId,
      selectedEdgeIndex: this.internalState.selectedEdgeIndex,
      highlightedEdgeId: this.internalState.highlightedEdgeId,
      highlightedEdgeIndex: this.internalState.highlightedEdgeIndex,
      selectedSourceNodeId: this.internalState.selectedSourceNodeId,
      selectedTargetNodeId: this.internalState.selectedTargetNodeId,
      highlightedSourceNodeId: this.internalState.highlightedSourceNodeId,
      highlightedTargetNodeId: this.internalState.highlightedTargetNodeId,
    };
  }
  draw({uniforms, ...rest}: {uniforms?: any, context?: DeckContext} = {}) {
    this.state.model.draw({
      ...rest,
      uniforms: {
        ...uniforms,
        highlightedNode: this.props.highlightedNode,
        highlightedEdge: this.props.highlightedEdge,
        strokeWidth: Math.max(1, this.props.width || 1),
        opacity: Math.max(0, Math.min(this.props.opacity, 1)),
      }
    });
  }
  getPickingInfo({mode, info}: {info: PickingInfo, mode: 'hover'|'click'}) {
    if (info.index === -1) {
      info.edgeId       = -1;
      info.sourceNodeId = -1;
      info.targetNodeId = -1;
    } else if (info.index === this.internalState.highlightedEdgeIndex) {
      info.edgeId       = this.internalState.highlightedEdgeIndex;
      info.sourceNodeId = this.internalState.highlightedSourceNodeId;
      info.targetNodeId = this.internalState.highlightedTargetNodeId;
    } else {
      info.edgeId                = info.index;
      const {buffer, offset = 0} = this.props.data.attributes.instanceEdges;
      ([info.sourceNodeId, info.targetNodeId] = buffer.getData({
        length: 2,
        srcByteOffset: <number>offset + (info.index * buffer.accessor.BYTES_PER_VERTEX),
      }));
    }
    this.internalState.highlightedEdgeId       = info.edgeId;
    this.internalState.highlightedEdgeIndex    = info.index;
    this.internalState.highlightedSourceNodeId = info.sourceNodeId;
    this.internalState.highlightedTargetNodeId = info.targetNodeId;
    if (mode === 'click') {
      this.internalState.selectedEdgeId       = this.internalState.highlightedEdgeId;
      this.internalState.selectedEdgeIndex    = this.internalState.highlightedEdgeIndex;
      this.internalState.selectedSourceNodeId = this.internalState.highlightedSourceNodeId;
      this.internalState.selectedTargetNodeId = this.internalState.highlightedTargetNodeId;
    }
    info.object = info.index;  // deck.gl uses info.object to check if item has already been added
    return info;
  }
  _getModel({gl, shaderCache}: DeckContext) {
    return new Model(gl, <any>{
      id: this.props.id,
      shaderCache,
      modules: [project32, picking],
      vs: edgeVertexShader,
      fs: edgeFragmentShader,
      uniforms: {numSegments: NUM_SEGMENTS},
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
            Array.from({length: NUM_SEGMENTS}).flatMap((_, i) => [i, -1, 0, i, 1, 0]))
        },
      }),
    });
  }
}
