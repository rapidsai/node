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

import {DeckCompositeLayer, DeckContext, UpdateStateProps} from '../deck.gl';

const {CompositeLayer} = require('@deck.gl/core');

import {Accessor} from '@luma.gl/webgl';
import {Buffer} from '../buffer';

import {PointCloudGPUBase} from './point-cloud/point-cloud-base';
import {PointColorBuffer, PointPositionBuffer} from './point-cloud/attributes';

const pointBufferNames = ['pointPositionX', 'pointPositionY', 'pointPositionZ', 'pointColor'];

export class PointCloudLayer extends (CompositeLayer as typeof DeckCompositeLayer) {
  static get layerName() { return 'PointCloudLayer'; }
  static get defaultProps() {
    return {
      numPoints: {type: 'number', min: 0, value: 0},
      ...PointCloudGPUBase.defaultProps,
    };
  }
  static getAccessors(context: DeckContext) { return PointCloudGPUBase.getAccessors(context); }

  initializeState({gl}: DeckContext) {
    this.setState({
      numPointsLoaded: 0,
      buffers: {
        // points
        pointPositionX: new PointPositionBuffer(gl),
        pointPositionY: new PointPositionBuffer(gl),
        pointPositionZ: new PointPositionBuffer(gl),
        pointColor: new PointColorBuffer(gl),
      },
    });
  }

  shouldUpdateState({props, oldProps, changeFlags, ...rest}: UpdateStateProps) {
    return changeFlags.viewportChanged ||
           super.shouldUpdateState({props, changeFlags, oldProps, ...rest});
  }
  updateState({props, oldProps, changeFlags, ...rest}: UpdateStateProps) {
    super.updateState({props, oldProps, changeFlags, ...rest});

    changeFlags = {
      ...changeFlags,
      pointsChanged: false,
      numPointsChanged: false,
    };

    const updates = [];

    if (changeFlags.dataChanged && props.data) {
      updates[0]                = props.data;
      changeFlags.pointsChanged = updates.some((x) => !!x.points && x.points.length > 0);
    }
    if (changeFlags.propsChanged) {
      changeFlags.numPointsChanged =
        props.numPoints > 0 && (oldProps.numPoints !== props.numPoints);
    }

    if (changeFlags.numPointsChanged) {
      resizeBuffers(props.numPoints, pointBufferNames.map((name) => this.state.buffers[name]));
    }

    if (changeFlags.pointsChanged) {
      this.setState(copyUpdatesIntoBuffers({...this.state, updates}));
    }
  }

  serialize() { return {}; }

  finalizeState(context?: DeckContext) {
    [...pointBufferNames]
      .map((name) => this.state.buffers[name])
      .filter(Boolean)
      .forEach((b) => b.delete());

    return super.finalizeState(context);
  }

  renderLayers() {
    const layers: any    = [];
    const {props, state} = this;
    const maxNumElements = 16777215 / 3;
    const renderChunks = (numElements: number,
                          LayerClass: any,
                          getProps: (index: number, offset: number, length: number) => any) => {
      const count = Math.ceil(numElements / maxNumElements);
      for (let index = -1; ++index < count;) {
        const offset = (index * maxNumElements);
        const length = Math.min(maxNumElements, numElements - offset);
        layers.push(new LayerClass(this.getSubLayerProps(getProps(index, offset, length))));
      }
      console.log(layers, count);
    };

    renderChunks(this.state.numPointsLoaded,
                 PointCloudGPUBase,
                 (index, offset) => ({
                   id: `${PointCloudGPUBase.layerName}-${index}`,
                   ...PointCloudGPUBaseProps(props, state, offset, length),
                 }));
    // console.log(state.numPointsLoaded);

    // layers.push(new PointCloudGPUBase(
    //   this.getSubLayerProps(PointCloudGPUBaseProps(props, state, 0, state.numPointsLoaded))));
    return layers;
  }
}

type LumaBuffer    = import('@luma.gl/webgl').Buffer;
const resizeBuffer = (length: number, buffer: LumaBuffer) =>
  buffer.reallocate(length * (buffer.accessor as Accessor).BYTES_PER_VERTEX);

const resizeBuffers = (length: number, buffers: LumaBuffer[]) =>
  buffers.forEach((buffer) => resizeBuffer(length, buffer));

const copyIntoBuffer = (target: LumaBuffer, source: any, offset: number) => target.subData({
  data: source,
  srcOffset: source.byteOffset,
  offset: offset * (target.accessor as Accessor).BYTES_PER_VERTEX
});

const copyUpdatesIntoBuffers = ({buffers, updates, numPointsLoaded}: any) => {
  const updatedBufferNames = (names: string[], {attributes = {}}: any) =>
    names.filter((name) => attributes[name]);
  const copyUpdateIntoBuffers = (buffers: any, names: string[], update: any) => {
    for (const name of updatedBufferNames(names, update)) {
      copyIntoBuffer(buffers[name], update.attributes[name], update.offset);
    }
  };

  const buffersToUpdate = [
    ...updates.reduce((names: string[], {points = {}}: any) => new Set([
                        ...names,
                        ...updatedBufferNames(pointBufferNames, points),
                      ]),
                      new Set())
  ].map((name) => buffers[name]);

  Buffer.mapResources(buffersToUpdate);

  updates.forEach(({points = {}}: any) => {
    points.length > 0 && copyUpdateIntoBuffers(buffers, pointBufferNames, points);
    numPointsLoaded = Math.max(numPointsLoaded, (<number>points.length) || 0);
  });

  Buffer.unmapResources(buffersToUpdate);

  return {numPointsLoaded};
};

const sliceLayerAttrib = (multiplier: number, buffer: LumaBuffer, offset = 0) =>
  ({buffer, offset: (buffer.accessor as Accessor).BYTES_PER_VERTEX * multiplier + offset});

const PointCloudGPUBaseProps = (props: any, state: any, offset: number, length: number) => ({
  sizeUnits: props.sizeUnits,
  numInstances: length,
  getNormal: props.getNormal,
  getColor: props.getColor,
  data: {
    attributes: {
      instancePositionsX: sliceLayerAttrib(offset, state.buffers.pointPositionX),
      instancePositionsY: sliceLayerAttrib(offset, state.buffers.pointPositionY),
      instancePositionsZ: sliceLayerAttrib(offset, state.buffers.pointPositionZ),
      instancePositionsX64Low: sliceLayerAttrib(offset, state.buffers.pointPositionX),
      instancePositionsY64Low: sliceLayerAttrib(offset, state.buffers.pointPositionY),
      instancePositionsZ64Low: sliceLayerAttrib(offset, state.buffers.pointPositionZ),
      instanceColors: sliceLayerAttrib(offset, state.buffers.pointColor)
    }
  },
});
