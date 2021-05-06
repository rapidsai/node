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

import { CompositeLayer } from '@deck.gl/core';
import { Buffer, Texture2D, Transform } from '@luma.gl/core';
import GL from '@luma.gl/constants';

import { getLayerAttributes } from './utils';
import NodeLayer from './node/node-layer';
import EdgeLayer from './edge/bezier-curve-layer';

import edgePositionsVS from './edge/edge-positions-vertex.glsl';

import { Uint8Buffer, CUDA } from '@nvidia/cuda';

const defaultProps = {
  numNodes: 0,
  numEdges: 0,
  nodeUpdates: [],
  edgeUpdates: [],
  drawEdges: true,
  edgeWidth: 1,
  edgeOpacity: 0.1,
};

const TEXTURE_WIDTH = 256;
const nodeLayerAttributes = getLayerAttributes(NodeLayer);
const edgeLayerAttributes = getLayerAttributes(EdgeLayer);

/*
LayerAttribute.allocate(numInstances) also creates a typed array as `value`, which we don't want.
Maybe extend it into
LayerAttribute.allocate(numInstances, {valueArray = true})
*/
function resizeBuffer(buffer, numInstances) {
  if (buffer.byteLength !== (numInstances * buffer.accessor.BYTES_PER_VERTEX)) {
    buffer.reallocate(numInstances * buffer.accessor.BYTES_PER_VERTEX);
  }
}

function registerCUDAGraphicsResources(webGLToCUDABufferMap, cudaResourceToBuffersMap, webGLBuffers) {
  webGLBuffers.forEach((glBuffer) => {
    if (!webGLToCUDABufferMap.has(glBuffer) || !webGLToCUDABufferMap.get(glBuffer)[1]) {
      try {
        const cuGraphicsResource = CUDA.runtime.cudaGraphicsGLRegisterBuffer(glBuffer.handle.ptr, 0);
        webGLToCUDABufferMap.set(glBuffer, [cuGraphicsResource, null]);
        cudaResourceToBuffersMap.set(cuGraphicsResource, [glBuffer, null]);
      } catch (e) { }
    }
  });
}

function getCUDAGraphicsResourcesBuffers(webGLToCUDABufferMap, cudaResourceToBuffersMap, webGLBuffers) {
  webGLBuffers.forEach((glBuffer) => {
    if (!webGLToCUDABufferMap.has(glBuffer) || !webGLToCUDABufferMap.get(glBuffer)[1]) {
      const [cuGraphicsResource] = webGLToCUDABufferMap.get(glBuffer);
      try {
        const cuBuffer = new Uint8Buffer(CUDA.runtime.cudaGraphicsResourceGetMappedPointer(cuGraphicsResource));
        webGLToCUDABufferMap.set(glBuffer, [cuGraphicsResource, cuBuffer]);
        cudaResourceToBuffersMap.set(cuGraphicsResource, [glBuffer, cuBuffer]);
      } catch (e) { }
    }
  });
}

function mapCUDAGraphicsResources(webGLToCUDABufferMap, cudaResourceToBuffersMap, webGLBuffers) {
  registerCUDAGraphicsResources(webGLToCUDABufferMap, cudaResourceToBuffersMap, webGLBuffers);
  CUDA.runtime.cudaGraphicsMapResources([...cudaResourceToBuffersMap.keys()]);
  getCUDAGraphicsResourcesBuffers(webGLToCUDABufferMap, cudaResourceToBuffersMap, webGLBuffers);
  return [webGLToCUDABufferMap, cudaResourceToBuffersMap];
}

function unmapCUDAGraphicsResources(webGLToCUDABufferMap, cudaResourceToBuffersMap, webGLBuffers) {
  CUDA.runtime.cudaGraphicsUnmapResources([...cudaResourceToBuffersMap.keys()]);
  webGLBuffers.forEach((glBuffer) => {
    if (webGLToCUDABufferMap.has(glBuffer) && webGLToCUDABufferMap.get(glBuffer)[1]) {
      const [cuGraphicsResource] = webGLToCUDABufferMap.get(glBuffer);
      webGLToCUDABufferMap.delete(glBuffer);
      cudaResourceToBuffersMap.delete(cuGraphicsResource);
    }
  });
}

/*
Always use bufferSubData in
LayerAttribute.updateBuffer ?
*/
function updatePartialBuffer(buffer, data, instanceOffset, webGLToCUDABufferMap) {
  const [, cuBuffer] = webGLToCUDABufferMap.get(buffer);
  try {
    cuBuffer.copyFrom(data, instanceOffset * buffer.accessor.BYTES_PER_VERTEX);
  } catch (e) { }
}

export default class ArrowGraphLayer extends CompositeLayer {
  static get layerName() { return 'ArrowGraphLayer'; }
  initializeState() {
    const { gl } = this.context;
    this.setState({
      loadedNodeCount: 0,
      loadedEdgeCount: 0,
      hasRenderedEdges: false,
      webGLToCUDABufferMap: new Map(),
      cudaResourceToBuffersMap: new Map(),
      edgePositionsToUpdate: Object.create(null),

      // Node layer buffers
      nodeColorsBuffer: new Buffer(gl, {
        accessor: nodeLayerAttributes.instanceFillColors,
        byteLength: 1
      }),
      nodeRadiusBuffer: new Buffer(gl, {
        accessor: nodeLayerAttributes.instanceRadius,
        byteLength: 1
      }),
      nodePositionsBuffer: new Buffer(gl, {
        accessor: nodeLayerAttributes.instancePositions,
        byteLength: 1
      }),

      // Line layer buffers
      edgeSourcePositionsBuffer: new Buffer(gl, {
        accessor: edgeLayerAttributes.instanceSourcePositions,
        byteLength: 1
      }),
      edgeTargetPositionsBuffer: new Buffer(gl, {
        accessor: edgeLayerAttributes.instanceTargetPositions,
        byteLength: 1
      }),
      edgeControlPointsBuffer: new Buffer(gl, {
        accessor: edgeLayerAttributes.instanceTargetPositions,
        byteLength: 1
      }),
      edgeColorsBuffer: new Buffer(gl, {
        accessor: { ...edgeLayerAttributes.instanceSourceColors, size: 8 },
        byteLength: 1
      }),

      // Transform feedback buffers
      nodePositionsTexture: new Texture2D(gl, {
        format: GL.RG32F,
        type: GL.FLOAT,
        width: 1,
        height: 1,
        parameters: {
          [GL.TEXTURE_MIN_FILTER]: [GL.NEAREST],
          [GL.TEXTURE_MAG_FILTER]: [GL.NEAREST]
        },
        mipmap: false
      }),
      edgesBuffer: new Buffer(gl, {
        accessor: { type: GL.UNSIGNED_INT, size: 2 },
        byteLength: 1
      }),
      edgeBundlesBuffer: new Buffer(gl, {
        accessor: { type: GL.UNSIGNED_INT, size: 2 },
        byteLength: 1
      }),

      edgeSourcePositionsBufferTemp: new Buffer(gl, {
        accessor: edgeLayerAttributes.instanceSourcePositions,
        byteLength: 1
      }),
      edgeTargetPositionsBufferTemp: new Buffer(gl, {
        accessor: edgeLayerAttributes.instanceTargetPositions,
        byteLength: 1
      }),
      edgeControlPointsBufferTemp: new Buffer(gl, {
        accessor: edgeLayerAttributes.instanceTargetPositions,
        byteLength: 1
      }),
    });

    this.setState({
      // Transform feedback that looks up node positions from ids
      edgePositionsTransform: new Transform(gl, {
        sourceBuffers: {
          edge: this.state.edgesBuffer,
          bundle: this.state.edgeBundlesBuffer,
          // instanceIds: this.state.edgesBuffer,
        },
        feedbackBuffers: {
          controlPoints: this.state.edgeControlPointsBufferTemp,
          sourcePositions: this.state.edgeSourcePositionsBufferTemp,
          targetPositions: this.state.edgeTargetPositionsBufferTemp
        },
        vs: edgePositionsVS,
        varyings: ['sourcePositions', 'targetPositions', 'controlPoints'],
        elementCount: 1,
        isInstanced: false,
      })
    });
  }

  /* eslint-disable max-statements */
  updateState({ props, oldProps }) {
    const { nodeUpdates, edgeUpdates, numNodes, numEdges, drawEdges } = props;
    const {
      nodeColorsBuffer,
      nodeRadiusBuffer,
      edgeBundlesBuffer,
      nodePositionsBuffer,
      nodePositionsTexture,
      edgePositionsTransform,
      edgeControlPointsBuffer,
      edgeSourcePositionsBuffer,
      edgeTargetPositionsBuffer,
      edgeControlPointsBufferTemp,
      edgeSourcePositionsBufferTemp,
      edgeTargetPositionsBufferTemp,
      edgeColorsBuffer,
      edgesBuffer,
      webGLToCUDABufferMap, cudaResourceToBuffersMap
    } = this.state;

    let { hasRenderedEdges, loadedNodeCount, loadedEdgeCount } = this.state;

    // Resize node layer buffers
    if (numNodes && numNodes !== oldProps.numNodes) {
      resizeBuffer(nodeColorsBuffer, numNodes);
      resizeBuffer(nodeRadiusBuffer, numNodes);
      nodePositionsTexture.resize({ width: TEXTURE_WIDTH, height: Math.ceil(numNodes / TEXTURE_WIDTH) });
      resizeBuffer(nodePositionsBuffer, nodePositionsTexture.width * nodePositionsTexture.height);
      loadedNodeCount = 0;
    }

    // Resize edge layer buffers
    if (numEdges && numEdges !== oldProps.numEdges) {
      resizeBuffer(edgeBundlesBuffer, numEdges);
      resizeBuffer(edgeControlPointsBuffer, numEdges);
      resizeBuffer(edgeSourcePositionsBuffer, numEdges);
      resizeBuffer(edgeTargetPositionsBuffer, numEdges);
      resizeBuffer(edgeColorsBuffer, numEdges);
      resizeBuffer(edgesBuffer, numEdges);
      loadedEdgeCount = 0;
    }

    const nodesUpdated = nodeUpdates.length > 0;
    const edgesUpdated = edgeUpdates.length > 0;
    const webglBuffers = [edgesBuffer, edgeColorsBuffer, nodeColorsBuffer, edgeBundlesBuffer, nodeRadiusBuffer, nodePositionsBuffer];

    (nodesUpdated || edgesUpdated) && mapCUDAGraphicsResources(webGLToCUDABufferMap, cudaResourceToBuffersMap, webglBuffers);

    // Apply node data updates
    while (nodeUpdates.length) {
      const { length, offset, color, size, position } = nodeUpdates.shift();
      color && updatePartialBuffer(nodeColorsBuffer, color, offset, webGLToCUDABufferMap);
      size && updatePartialBuffer(nodeRadiusBuffer, size, offset, webGLToCUDABufferMap);
      position && updatePartialBuffer(nodePositionsBuffer, position, offset, webGLToCUDABufferMap);
      loadedNodeCount = Math.max(loadedNodeCount, offset + length);
    }

    // Apply edge data updates
    let edgePositionsToUpdate = this.state.edgePositionsToUpdate;
    while (edgeUpdates.length) {
      const { length, offset, edge, color, bundle } = edgeUpdates.shift();
      edge && updatePartialBuffer(edgesBuffer, edge, offset, webGLToCUDABufferMap);
      color && updatePartialBuffer(edgeColorsBuffer, color, offset, webGLToCUDABufferMap);
      bundle && updatePartialBuffer(edgeBundlesBuffer, bundle, offset, webGLToCUDABufferMap);
      loadedEdgeCount = Math.max(loadedEdgeCount, offset + length);
      edgePositionsToUpdate[`[${offset},${length}]`] = { offset, length };
    }

    (nodesUpdated || edgesUpdated) && unmapCUDAGraphicsResources(webGLToCUDABufferMap, cudaResourceToBuffersMap, webglBuffers);

    // Update edge position buffers
    if (drawEdges && numEdges > 0) {

      const allNodesLoaded = (numNodes > 0 && loadedNodeCount === numNodes
        && (nodeColorsBuffer.byteLength / nodeColorsBuffer.accessor.BYTES_PER_VERTEX) === numNodes
        && (nodeRadiusBuffer.byteLength / nodeRadiusBuffer.accessor.BYTES_PER_VERTEX) === numNodes
      );

      if (!hasRenderedEdges || (nodesUpdated && allNodesLoaded)) {
        nodePositionsTexture.setImageData({ data: nodePositionsBuffer });
      }

      const allEdgesLoaded = (numEdges > 0 && loadedEdgeCount === numEdges
        && (edgesBuffer.byteLength / edgesBuffer.accessor.BYTES_PER_VERTEX) === numEdges
        && (edgeColorsBuffer.byteLength / edgeColorsBuffer.accessor.BYTES_PER_VERTEX) === numEdges
        && (edgeBundlesBuffer.byteLength / edgeBundlesBuffer.accessor.BYTES_PER_VERTEX) === numEdges
      );

      const edgePositionUpdateInfo = {
        loadedNodeCount,
        loadedEdgeCount,
        nodePositionsTexture,
        edgePositionsTransform,
        edgeControlPointsBuffer,
        edgeSourcePositionsBuffer,
        edgeTargetPositionsBuffer,
        edgeControlPointsBufferTemp,
        edgeSourcePositionsBufferTemp,
        edgeTargetPositionsBufferTemp,
        edgeWidth: this.props.edgeWidth,
      };

      if (allEdgesLoaded && (nodesUpdated || !hasRenderedEdges)) {
        hasRenderedEdges = true;
        edgePositionUpdateInfo.offset = 0;
        edgePositionUpdateInfo.length = numEdges;
        copyEdgePositions(edgePositionUpdateInfo);
      } else if (allNodesLoaded && (edgesUpdated || !hasRenderedEdges)) {
        hasRenderedEdges = true;
        for (const k in edgePositionsToUpdate) {
          edgePositionUpdateInfo.offset = edgePositionsToUpdate[k].offset;
          edgePositionUpdateInfo.length = edgePositionsToUpdate[k].length;
          copyEdgePositions(edgePositionUpdateInfo);
        }
        edgePositionsToUpdate = Object.create(null);
      }
    }

    this.setState({ loadedNodeCount, loadedEdgeCount, hasRenderedEdges, edgePositionsToUpdate });
  }
  /* eslint-enable max-statements */

  renderLayers() {
    const {
      loadedNodeCount,
      loadedEdgeCount,
      nodeColorsBuffer,
      nodeRadiusBuffer,
      nodePositionsBuffer,
      edgeControlPointsBuffer,
      edgeSourcePositionsBuffer,
      edgeTargetPositionsBuffer,
      edgeColorsBuffer
    } = this.state;

    return [
      loadedEdgeCount &&
      new EdgeLayer(
        this.getSubLayerProps({
          id: 'edges',
          numInstances: loadedEdgeCount,
          visible: this.props.drawEdges,
          opacity: this.props.edgeOpacity,
          strokeWidth: this.props.edgeWidth,
          data: {
            attributes: {
              instanceSourcePositions: edgeSourcePositionsBuffer,
              instanceTargetPositions: edgeTargetPositionsBuffer,
              instanceControlPoints: edgeControlPointsBuffer,
              instanceSourceColors: edgeColorsBuffer,
              instanceTargetColors: edgeColorsBuffer,
            }
          },
          pickable: true,
          autoHighlight: true,
          highlightColor: [255, 255, 255, 255],
        })
      ),
      loadedNodeCount &&
      new NodeLayer(
        this.getSubLayerProps({
          id: 'nodes',
          numInstances: loadedNodeCount,
          data: {
            attributes: {
              instanceRadius: nodeRadiusBuffer,
              instanceFillColors: nodeColorsBuffer,
              instanceLineColors: nodeColorsBuffer,
              instancePositions: nodePositionsBuffer,
            }
          },
          opacity: 0.5,
          radiusScale: 1,
          radiusMinPixels: 0,
          radiusMaxPixels: 25,
          // interaction
          pickable: true,
          autoHighlight: true,
          highlightColor: [255, 255, 255, 255],
        })
      )
    ];
  }
}

ArrowGraphLayer.defaultProps = defaultProps;

function copyEdgePositions({
  offset,
  length,
  edgeWidth,
  loadedNodeCount,
  loadedEdgeCount,
  // nodeXTexture,
  // nodeYTexture,
  nodePositionsTexture,
  edgePositionsTransform,
  edgeControlPointsBuffer,
  edgeSourcePositionsBuffer,
  edgeTargetPositionsBuffer,
  edgeControlPointsBufferTemp,
  edgeSourcePositionsBufferTemp,
  edgeTargetPositionsBufferTemp
}) {

  if (length <= 0) return;

  // Update edge position buffers
  resizeBuffer(edgeControlPointsBufferTemp, length);
  resizeBuffer(edgeSourcePositionsBufferTemp, length);
  resizeBuffer(edgeTargetPositionsBufferTemp, length);
  edgePositionsTransform.update({ elementCount: length, });
  edgePositionsTransform.run({
    offset, uniforms: {
      loadedNodeCount,
      loadedEdgeCount,
      width: TEXTURE_WIDTH,
      strokeWidth: edgeWidth,
      nodePositions: nodePositionsTexture
    }
  });

  edgeControlPointsBuffer.copyData({
    sourceBuffer: edgeControlPointsBufferTemp,
    size: length * edgeControlPointsBuffer.accessor.BYTES_PER_VERTEX,
    writeOffset: offset * edgeControlPointsBuffer.accessor.BYTES_PER_VERTEX,
  });

  edgeSourcePositionsBuffer.copyData({
    sourceBuffer: edgeSourcePositionsBufferTemp,
    size: length * edgeSourcePositionsBuffer.accessor.BYTES_PER_VERTEX,
    writeOffset: offset * edgeSourcePositionsBuffer.accessor.BYTES_PER_VERTEX,
  });

  edgeTargetPositionsBuffer.copyData({
    sourceBuffer: edgeTargetPositionsBufferTemp,
    size: length * edgeTargetPositionsBuffer.accessor.BYTES_PER_VERTEX,
    writeOffset: offset * edgeTargetPositionsBuffer.accessor.BYTES_PER_VERTEX,
  });

  resizeBuffer(edgeControlPointsBufferTemp, 0);
  resizeBuffer(edgeSourcePositionsBufferTemp, 0);
  resizeBuffer(edgeTargetPositionsBufferTemp, 0);
}
