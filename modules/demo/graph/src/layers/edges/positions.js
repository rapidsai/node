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
import { getParameters } from '@luma.gl/core';
import { EdgeComponentBuffer, EdgePositionTexture } from './attributes';
import computePositionVertexShader from './compute-position-vertex.glsl';

export class ComputeEdgePositionsTransform extends Transform {
    constructor(gl) {
        super(gl, {
            elementCount: 0,
            isInstanced: false,
            vs: computePositionVertexShader,
            uniforms: {
                textureWidth: 1,
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
    getTextureWidth(size) {
        return Math.min((size + 7) & ~7, getParameters(this.gl, this.gl.MAX_TEXTURE_SIZE));
    }
    roundSizeUpToTextureDimensions(size) {
        const length = (size + 7) & ~7;
        const width = this.getTextureWidth(length);
        return width * Math.ceil(length / width); // width * height;
    }
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

        const textureWidth = this.getTextureWidth(numNodes);
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
        resizeTexture(nodeXPositionsTexture, textureWidth, numNodes);
        resizeTexture(nodeYPositionsTexture, textureWidth, numNodes);
        // copy x and y node positions into place
        if (nodesChanged) {
            setSubImageData(nodeXPositionsTexture, nodeXPositions, offset);
            setSubImageData(nodeYPositionsTexture, nodeYPositions, offset);
        }

        this.update({
            elementCount: length,
            sourceBuffers: { edge: edgeList, bundle: edgeBundles }
        });
        this.run({ offset, uniforms: { numNodesLoaded, strokeWidth, textureWidth } });

        copyDtoD(edgeControlPoints, internalControlPoints, offset, length);
        copyDtoD(edgeSourcePositions, internalSourcePositions, offset, length);
        copyDtoD(edgeTargetPositions, internalTargetPositions, offset, length);
    }
}

const resizeBuffer = (length, buffer) => buffer.reallocate(length * buffer.accessor.BYTES_PER_VERTEX);
const resizeTexture = (texture, width, length) => {
    const height = Math.ceil(length / width);
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
