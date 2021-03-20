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

import '../deck.gl';

import {DeckContext, Layer, picking, PickingInfo, project32, UpdateStateProps} from '@deck.gl/core';
import {Geometry, Model} from '@luma.gl/engine';

import {
  nodeColorAccessor,
  nodeElementIndicesAccessor,
  nodePositionAccessor,
  nodeRadiusAccessor,
} from './nodes/attributes';
import nodeFragmentShader from './nodes/node-fragment.glsl';
import nodeVertexShader from './nodes/node-vertex.glsl';

export class NodeLayer extends Layer {
  static get layerName() { return 'NodeLayer'; }
  static get defaultProps() {
    return {
      filled: true,
      stroked: true,
      strokeRatio: 0.05,
      fillOpacity: 1,
      strokeOpacity: 1,
      radiusScale: 1,
      lineWidthScale: 1,
      //  min point radius in pixels
      radiusMinPixels: 0,
      // max point radius in pixels
      radiusMaxPixels: Number.MAX_SAFE_INTEGER,
      lineWidthMinPixels: 0,
      lineWidthMaxPixels: Number.MAX_SAFE_INTEGER,
      highlightedNode: -1,
      highlightedSourceNode: -1,
      highlightedTargetNode: -1,
    };
  }
  static getAccessors({gl}: {gl: WebGLRenderingContext}) {
    return {
      instanceRadius: {...nodeRadiusAccessor(gl), accessor: 'getRadius'},
      instanceFillColors: {...nodeColorAccessor(gl), accessor: 'getFillColor'},
      instanceLineColors: {...nodeColorAccessor(gl), accessor: 'getLineColor'},
      instanceXPositions: {...nodePositionAccessor(gl), accessor: 'getXPosition'},
      instanceYPositions: {...nodePositionAccessor(gl), accessor: 'getYPosition'},
      instanceNodeIndices: {...nodeElementIndicesAccessor(gl), accessor: 'getNodeIndex'},
      elementIndices:
        {...nodeElementIndicesAccessor(gl), accessor: 'getElementIndex', isIndexed: true},
    };
  }
  initializeState(context: DeckContext) {
    this.internalState.selectedNodeId       = -1;
    this.internalState.highlightedNodeId    = -1;
    this.internalState.selectedNodeIndex    = -1;
    this.internalState.highlightedNodeIndex = -1;
    this.getAttributeManager().addInstanced(NodeLayer.getAccessors(context));
  }
  updateState({props, oldProps, context, changeFlags}: UpdateStateProps) {
    super.updateState({props, oldProps, context, changeFlags});
    if (changeFlags.extensionsChanged) {
      if (this.state.model) { this.state.model.delete(); }
      this.setState({model: this._getModel(context)});
      this.getAttributeManager().invalidateAll();
    }
  }
  draw({uniforms, ...rest}: {uniforms?: any, context?: DeckContext} = {}) {
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
        highlightedNode: this.props.highlightedNode,
        highlightedSourceNode: this.props.highlightedSourceNode,
        highlightedTargetNode: this.props.highlightedTargetNode,
        ...uniforms,
      }
    });
  }
  getPickingInfo({mode, info}: {info: PickingInfo, mode: 'hover'|'click'}) {
    if (info.index === -1) {
      info.nodeId = info.index;
    } else if (this.internalState.highlightedNodeIndex === info.index) {
      info.nodeId = this.internalState.highlightedNodeId;
    } else {
      const {buffer, offset = 0} = this.props.data.attributes.instanceNodeIndices;
      ([info.nodeId] = buffer.getData({
        length: 1,
        srcByteOffset: <number>offset + (info.index * buffer.accessor.BYTES_PER_VERTEX),
      }));
    }
    this.internalState.highlightedNodeId    = info.nodeId;
    this.internalState.highlightedNodeIndex = info.index;
    if (mode === 'click') {
      this.internalState.selectedNodeId    = this.internalState.highlightedNodeId;
      this.internalState.selectedNodeIndex = this.internalState.highlightedNodeIndex;
    }
    return info;
  }
  _getModel({gl, shaderCache}: DeckContext) {
    return new Model(gl, <any>{
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
        attributes:
          {positions: {size: 3, value: new Float32Array([-1, -1, 0, -1, 1, 0, 1, 1, 0, 1, -1, 0])}}
      }),
    });
  }
}
