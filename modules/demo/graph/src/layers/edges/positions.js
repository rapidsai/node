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

import { Transform } from '@luma.gl/engine';
import { EdgeComponentBuffer, EdgePositionTexture } from './attributes';
import computePositionVertexShader from './compute-position-vertex.glsl';

export class ComputeEdgePositionsTransform extends Transform {
    constructor(gl) {
        super(gl, {
            elementCount: 0,
            isInstanced: false,
            vs: computePositionVertexShader,
            uniforms: {
                textureWidth: TEXTURE_WIDTH,
                nodeXPositions: new EdgePositionTexture(gl),
                nodeYPositions: new EdgePositionTexture(gl),
            },
            varyings: ['controlPoint', 'sourcePosition', 'targetPosition'],
            feedbackBuffers: {
                controlPoint: new EdgeComponentBuffer(gl, 1),
                sourcePosition: new EdgeComponentBuffer(gl, 1),
                targetPosition: new EdgeComponentBuffer(gl, 1),
            },
        });
    }
    get textureWidth() { return this.model.getUniforms().textureWidth; }
    call({
        offset = 0,
        length = 0,
        numNodes = 0,
        nodesChanged = false,
        numNodesLoaded = numNodes,
        strokeWidth = 1,
        edgeList,
        edgeBundles,
        nodeXPositions,
        nodeYPositions,
        edgeControlPoints,
        edgeSourcePositions,
        edgeTargetPositions,
    } = {}) {

        if (length <= 0) return;

        const internalControlPoints = this.getBuffer('controlPoint');
        const internalSourcePositions = this.getBuffer('sourcePosition');
        const internalTargetPositions = this.getBuffer('targetPosition');
        const {
            nodeXPositions: nodeXPositionsTexture,
            nodeYPositions: nodeYPositionsTexture,
        } = this.model.getUniforms();

        // resize internal edge component buffers
        resizeBuffer(length, internalControlPoints);
        resizeBuffer(length, internalSourcePositions);
        resizeBuffer(length, internalTargetPositions);
        // resize x and y node position textures
        resizeTexture(nodeXPositionsTexture, numNodes);
        resizeTexture(nodeYPositionsTexture, numNodes);
        // copy x and y node positions into place
        if (nodesChanged) {
            setSubImageData(nodeXPositionsTexture, nodeXPositions, offset);
            setSubImageData(nodeYPositionsTexture, nodeYPositions, offset);
        }

        this.update({
            elementCount: length,
            sourceBuffers: { edge: edgeList, bundle: edgeBundles }
        });
        this.run({ offset, uniforms: { numNodesLoaded, strokeWidth } });

        copyDtoD(edgeControlPoints, internalControlPoints, offset, length);
        copyDtoD(edgeSourcePositions, internalSourcePositions, offset, length);
        copyDtoD(edgeTargetPositions, internalTargetPositions, offset, length);
    }
}

const TEXTURE_WIDTH = 256;

const resizeBuffer = (length, buffer) => buffer.reallocate(length * buffer.accessor.BYTES_PER_VERTEX);
const resizeTexture = (texture, length) => {
    const width =  TEXTURE_WIDTH;
    const height = Math.ceil(length / TEXTURE_WIDTH);
    if (texture.width !== width || texture.height !== height) {
        texture.resize({ width, height });
    }
}

const setSubImageData = (texture, { handle: data, accessor, byteLength }, offset = 0) => texture.setSubImageData({
    data,
    offset: offset * accessor.BYTES_PER_VERTEX,
    height: Math.ceil((byteLength / accessor.BYTES_PER_VERTEX) / texture.width),
});

const copyDtoD = (target, source, offset, length) => target.copyData({
    sourceBuffer: source,
    size: length * target.accessor.BYTES_PER_VERTEX,
    writeOffset: offset * target.accessor.BYTES_PER_VERTEX,
});
