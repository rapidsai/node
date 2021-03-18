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

import { Buffer } from '@rapidsai/deck.gl';
import { TextLayer } from '@deck.gl/layers';
import { CompositeLayer } from '@deck.gl/core';

import { NodeLayer } from './nodes';
import { EdgeLayer } from './edges';
import { ComputeEdgePositionsTransform } from './edges/positions';
import { EdgeColorBuffer, EdgeListBuffer, EdgeComponentBuffer } from './edges/attributes';
import { NodeColorBuffer, NodeRadiusBuffer, NodePositionBuffer, NodeElementIndicesBuffer } from './nodes/attributes';

const edgeBufferNames = [
  'edgeList', 'edgeBundles', 'edgeColors', 'edgeControlPoints', 'edgeSourcePositions', 'edgeTargetPositions'
];
const nodeBufferNames = [
  'nodeXPositions', 'nodeYPositions', 'nodeRadius', 'nodeFillColors', 'nodeLineColors', 'nodeElementIndices'
];

export class GraphLayer extends CompositeLayer {
  static getAccessors(context) {
    return {
      ...NodeLayer.getAccessors(context),
      ...EdgeLayer.getAccessors(context),
    }
  }
  initializeState({ gl }) {
    // GPGPU
    this.internalState.computeEdgePositions = new ComputeEdgePositionsTransform(gl);
    this.setState({
      numNodesLoaded: 0,
      numEdgesLoaded: 0,
      highlightedEdge: -1,
      highlightedNode: -1,
      highlightedSourceNode: -1,
      highlightedTargetNode: -1,
      edgePositionRanges: new Set(),
      buffers: {
        // edges
        edgeList: new EdgeListBuffer(gl),
        edgeBundles: new EdgeListBuffer(gl),
        edgeColors: new EdgeColorBuffer(gl),
        edgeControlPoints: new EdgeComponentBuffer(gl),
        edgeSourcePositions: new EdgeComponentBuffer(gl),
        edgeTargetPositions: new EdgeComponentBuffer(gl),
        // nodes
        nodeRadius: new NodeRadiusBuffer(gl),
        nodeFillColors: new NodeColorBuffer(gl),
        nodeLineColors: new NodeColorBuffer(gl),
        nodeXPositions: new NodePositionBuffer(gl),
        nodeYPositions: new NodePositionBuffer(gl),
        nodeElementIndices: new NodeElementIndicesBuffer(gl),
      },
    });
  }
  shouldUpdateState({ props, oldProps, changeFlags, ...rest }) {
    return changeFlags.viewportChanged || super.shouldUpdateState({ props, changeFlags, ...rest });
  }
  updateState({ props, oldProps, changeFlags }) {

    changeFlags = {
      ...changeFlags,
      edgesChanged: false,
      nodesChanged: false,
      graphChanged: false,
      numEdgesChanged: false,
      numNodesChanged: false,
    };

    let updates = [];

    if (changeFlags.dataChanged && props.data) {
      updates[0] = props.data;
      changeFlags.edgesChanged = updates.some((x) => !!x.edges && x.edges.length > 0);
      changeFlags.nodesChanged = updates.some((x) => !!x.nodes && x.nodes.length > 0);
      changeFlags.graphChanged = changeFlags.edgesChanged || changeFlags.nodesChanged;
    }

    if (changeFlags.propsChanged) {
      changeFlags.numEdgesChanged = props.numEdges > 0 && (oldProps.numEdges !== props.numEdges);
      changeFlags.numNodesChanged = props.numNodes > 0 && (oldProps.numNodes !== props.numNodes);
    }

    if (changeFlags.numEdgesChanged) {
      resizeBuffers(props.numEdges, edgeBufferNames.map((name) => this.state.buffers[name]));
    }
    if (changeFlags.numNodesChanged) {
      nodeBufferNames.forEach((name) => {
        let length = props.numNodes;
        switch (name) {
          case 'nodeXPositions':
          case 'nodeYPositions':
            length = this.internalState.computeEdgePositions.roundSizeUpToTextureDimensions(length);
        }
        resizeBuffer(length, this.state.buffers[name]);
      });
    }
    if (changeFlags.graphChanged) {
      this.setState(copyUpdatesIntoBuffers({ ...this.state, updates }));
    }
    if (props.edgesVisible && props.numEdges > 0) {
      this.setState(computePendingEdgePositions({
        ...this.state, ...props, ...changeFlags,
        computeEdgePositions: this.internalState.computeEdgePositions,
        edgeBuffers: edgeBufferNames.map((name) => this.state.buffers[name]),
        nodeBuffers: nodeBufferNames.map((name) => this.state.buffers[name]),
      }));
    }
  }
  finalizeState() {

    [...edgeBufferNames, ...nodeBufferNames]
      .map((name) => this.state.buffers[name])
      .filter(Boolean).forEach((b) => b.delete());

    if (this.internalState.computeEdgePositions) {
      this.internalState.computeEdgePositions.delete();
      this.internalState.computeEdgePositions = null;
    }

    return super.finalizeState();
  }
  onHover({ layer, x, y, index, coordinate, edgeId = -1, nodeId = -1, sourceNodeId = -1, targetNodeId = -1 }) {
    this.setState({
      highlightedEdge: edgeId,
      highlightedNode: nodeId,
      highlightedSourceNode: sourceNodeId,
      highlightedTargetNode: targetNodeId,
      labelColor: [255, 255, 255],
      labelPosition: coordinate || layer.context.viewport.unproject([x, y]),
      labelText: ((names) => {
        const get = (path) => path.split('.').reduce(((xs, x) => xs ? xs[x] : null), this.props);
        return (nodeId !== -1) ?
          ((names = get('data.nodes.attributes.nodeName')) ? names.getValue(nodeId) : `${nodeId}`) :
          (edgeId !== -1) ?
            ((names = get('data.edges.attributes.edgeName')) ? names.getValue(edgeId) : `${sourceNodeId} - ${targetNodeId}`) : ``;
      })()
    });
  }
  renderLayers() {
    const layers = [];
    const { props, state } = this;
    const maxNumElements = 16777215 / 3;
    const renderChunks = (numElements, LayerClass, getProps) => {
      const count = Math.ceil(numElements / maxNumElements);
      for (let index = -1; ++index < count;) {
        const offset = (index * maxNumElements);
        const length = Math.min(maxNumElements, numElements - offset);
        layers.push(new LayerClass(this.getSubLayerProps(getProps(index, offset, length))));
      }
    };

    props.edgesVisible &&
      renderChunks(this.state.numEdgesLoaded, EdgeLayer, (index, offset, length) => ({
        id: `${props.id}-edge-layer-${index}`, ...edgeLayerProps(props, state, offset, length),
      }));

    // render bezier control points for debugging
    // renderChunks(this.state.numEdgesLoaded, ScatterplotLayer, (index, offset, length) => ({
    //     id: `${props.id}-bezier-control-points-layer-${index}`,
    //     numInstances: length,
    //     radiusScale: 2,
    //     data: {
    //         attributes: {
    //             instancePositions: { buffer: state.buffers.edgeControlPoints, offset: offset * 3 },
    //             instanceFillColors: { buffer: state.buffers.edgeColors, offset: offset * 4, stride: 8 },
    //         }
    //     },
    // }));

    props.nodesVisible &&
      renderChunks(this.state.numNodesLoaded, NodeLayer, (index, offset, length) => ({
        id: `${props.id}-node-layer-${index}`, ...nodeLayerProps(props, state, offset, length),
      }));

    if (this.state.labelText) {
      layers.push(new TextLayer(this.getSubLayerProps({
        id: `${props.id}-text-layer-0`, ...textLayerProps(this.props, this.state),
      })));
    }
    return layers;
  }
}

GraphLayer.layerName = 'GraphLayer';
GraphLayer.defaultProps = {
  // graph props
  numNodes: { type: 'number', min: 0, value: 0 },
  numEdges: { type: 'number', min: 0, value: 0 },
  nodesVisible: { type: 'boolean', value: true },
  edgesVisible: { type: 'boolean', value: true },
  // edge props
  edgeOpacity: EdgeLayer.defaultProps.opacity,
  edgeStrokeWidth: EdgeLayer.defaultProps.width,
  // node props
  nodesFilled: NodeLayer.defaultProps.filled,
  nodesStroked: NodeLayer.defaultProps.stroked,
  nodeFillOpacity: NodeLayer.defaultProps.fillOpacity,
  nodeStrokeRatio: NodeLayer.defaultProps.strokeRatio,
  nodeStrokeOpacity: NodeLayer.defaultProps.strokeOpacity,
  nodeRadiusScale: NodeLayer.defaultProps.radiusScale,
  nodeLineWidthScale: NodeLayer.defaultProps.lineWidthScale,
  nodeRadiusMinPixels: NodeLayer.defaultProps.radiusMinPixels,
  nodeRadiusMaxPixels: NodeLayer.defaultProps.radiusMaxPixels,
  nodeLineWidthMinPixels: NodeLayer.defaultProps.lineWidthMinPixels,
  nodeLineWidthMaxPixels: NodeLayer.defaultProps.lineWidthMaxPixels,
};

const resizeBuffer = (length, buffer) => buffer.reallocate(length * buffer.accessor.BYTES_PER_VERTEX);
const resizeBuffers = (length, buffers) => buffers.forEach((buffer) => resizeBuffer(length, buffer));

const copyIntoBuffer = (target, source, offset) => target.subData({
  data: source,
  srcOffset: source.byteOffset,
  offset: offset * target.accessor.BYTES_PER_VERTEX
});

const copyUpdatesIntoBuffers = ({
  buffers,
  updates,
  numEdgesLoaded,
  numNodesLoaded,
  edgePositionRanges,
}) => {

  const updatedBufferNames = (names, { attributes }) => names.filter((name) => attributes[name]);
  const copyUpdateIntoBuffers = (buffers, names, update) => {
    for (const name of updatedBufferNames(names, update)) {
      copyIntoBuffer(buffers[name], update.attributes[name], update.offset);
    }
  };

  const buffersToUpdate = [
    ...updates.reduce((names, { edges = {}, nodes = {} }) => new Set([
      ...names,
      ...updatedBufferNames(edgeBufferNames, edges),
      ...updatedBufferNames(nodeBufferNames, nodes),
    ]), new Set())
  ].map((name) => buffers[name]);

  Buffer.mapResources(buffersToUpdate);

  updates.forEach(({ edges = {}, nodes = {} }) => {
    edges.offset = Math.max(0, edges.offset || 0);
    nodes.offset = Math.max(0, nodes.offset || 0);
    edges.length > 0 && edgePositionRanges.add([edges.offset, edges.length]);
    edges.length > 0 && copyUpdateIntoBuffers(buffers, edgeBufferNames, edges);
    nodes.length > 0 && copyUpdateIntoBuffers(buffers, nodeBufferNames, nodes);
    numEdgesLoaded = Math.max(numEdgesLoaded, (edges.offset + edges.length) || 0);
    numNodesLoaded = Math.max(numNodesLoaded, (nodes.offset + nodes.length) || 0);
  });

  Buffer.unmapResources(buffersToUpdate);

  return { numEdgesLoaded, numNodesLoaded, edgePositionRanges };
};

const computePendingEdgePositions = ({
  buffers,
  numEdges,
  numNodes,
  edgeBuffers,
  nodeBuffers,
  nodesChanged,
  edgesChanged,
  numEdgesLoaded,
  numNodesLoaded,
  edgeStrokeWidth,
  edgePositionRanges,
  computeEdgePositions,
}) => {

  const allElementsLoaded = (numTotal, numLoaded, buffers) => (
    numTotal > 0 && numLoaded === numTotal && buffers.every((buffer) =>
      numTotal === (buffer.byteLength / buffer.accessor.BYTES_PER_VERTEX)));

  const computeEdgePositionRanges = (ranges) => {
    const args = {
      ...buffers, edgeStrokeWidth,
      offset: 0, length: numEdges,
      numNodes, numNodesLoaded, nodesChanged,
    };
    ranges.forEach(([offset, length]) => {
      args.offset = offset;
      args.length = length;
      computeEdgePositions.call(args);
    });
    return { edgePositionRanges: new Set() };
  };

  // If new node positions load after logical edges, copy all the
  // new node positions to the source and target edge positions.
  if (nodesChanged && allElementsLoaded(numEdges, numEdgesLoaded, edgeBuffers)) {
    return computeEdgePositionRanges(new Set([[0, numEdges]]));
  }

  if (edgesChanged && allElementsLoaded(numNodes, numNodesLoaded, nodeBuffers)) {
    return computeEdgePositionRanges([...edgePositionRanges].sort(([a], [b]) => a - b));
  }

  return { edgePositionRanges };
};

const sliceLayerAttrib = (multiplier, buffer, offset = 0) => ({
  buffer,
  offset: buffer.accessor.BYTES_PER_VERTEX * multiplier + offset
});

const edgeLayerProps = (props, state, offset, length) => ({
  pickable: true,
  autoHighlight: false,
  highlightColor: [225, 225, 225, 100],
  numInstances: length,
  opacity: props.edgeOpacity,
  visible: props.edgesVisible,
  width: props.edgeStrokeWidth,
  highlightedNode: state.highlightedNode,
  highlightedEdge: state.highlightedEdge,
  data: {
    attributes: {
      instanceEdges: sliceLayerAttrib(offset, state.buffers.edgeList),
      instanceSourceColors: sliceLayerAttrib(offset, state.buffers.edgeColors),
      instanceTargetColors: sliceLayerAttrib(offset, state.buffers.edgeColors, 4),
      instanceControlPoints: sliceLayerAttrib(offset, state.buffers.edgeControlPoints),
      instanceSourcePositions: sliceLayerAttrib(offset, state.buffers.edgeSourcePositions),
      instanceTargetPositions: sliceLayerAttrib(offset, state.buffers.edgeTargetPositions),
    }
  }
});

const nodeLayerProps = (props, state, offset, length) => ({
  pickable: true,
  autoHighlight: false,
  highlightColor: [225, 225, 225, 100],
  numInstances: length,
  filled: props.nodesFilled,
  stroked: props.nodesStroked,
  visible: props.nodesVisible,
  fillOpacity: props.nodeFillOpacity,
  strokeRatio: props.nodeStrokeRatio,
  strokeOpacity: props.nodeStrokeOpacity,
  radiusScale: props.nodeRadiusScale,
  radiusMinPixels: props.nodeRadiusMinPixels,
  radiusMaxPixels: props.nodeRadiusMaxPixels,
  lineWidthScale: props.nodeLineWidthScale,
  lineWidthMinPixels: props.nodeLineWidthMinPixels,
  lineWidthMaxPixels: props.nodeLineWidthMaxPixels,
  highlightedNode: state.highlightedNode,
  highlightedSourceNode: state.highlightedSourceNode,
  highlightedTargetNode: state.highlightedTargetNode,
  data: {
    attributes: {
      instanceRadius: sliceLayerAttrib(offset, state.buffers.nodeRadius),
      instanceFillColors: sliceLayerAttrib(offset, state.buffers.nodeFillColors),
      instanceLineColors: sliceLayerAttrib(offset, state.buffers.nodeFillColors),
      instanceXPositions: sliceLayerAttrib(offset, state.buffers.nodeXPositions),
      instanceYPositions: sliceLayerAttrib(offset, state.buffers.nodeYPositions),
      instanceXPositions64Low: sliceLayerAttrib(offset, state.buffers.nodeXPositions),
      instanceYPositions64Low: sliceLayerAttrib(offset, state.buffers.nodeYPositions),
      instanceNodeIndices: sliceLayerAttrib(offset, state.buffers.nodeElementIndices),
      elementIndices: sliceLayerAttrib(offset, state.buffers.nodeElementIndices),
    }
  },
});

const textLayerProps = (props, state) => ({
  sizeScale: 1.0,
  opacity: 1.0,
  maxWidth: -1,
  pickable: false,
  backgroundColor: [46, 46, 46],
  getTextAnchor: 'start',
  getAlignmentBaseline: 'top',
  fontFamily: 'sans-serif, sans',
  getSize: d => d.size,
  getColor: d => d.color,
  getPosition: d => d.position,
  getPixelOffset: d => [d.size, 0],
  data: [{
    size: 12,
    text: state.labelText,
    color: state.labelColor,
    position: state.labelPosition,
  }]
});
