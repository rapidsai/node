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
import { GPUPointInPolygon } from '@luma.gl/experimental';

const FILTER_VS = `\
#version 300 es
in float position_x;
in float position_y;
out float filterValueIndex; //[x: 0 (outside polygon)/1 (inside), y: position index]
void main()
{
    vec2 result = textureFilter_filter(vec2(position_x, position_y));
    // filterValueIndex = mix(result.x, result.y, step(0., result.x));
    filterValueIndex = result.x == -1. ? -1. : result.y;
}
`;

export class NodesInViewport extends GPUPointInPolygon {
    _setupResources() {
        super._setupResources();
        const { modules } = this.filterTransform.model.programProps;
        this.filterTransform.delete();
        this.filterTransform = new Transform(this.gl, {
            id: 'filter-points-in-view-transform',
            vs: FILTER_VS,
            modules: modules,
            varyings: ['filterValueIndex']
        });
    }
    filter({ xPosition, yPosition, filterValueIndexBuffer, count }) {
        this.filterTransform.update({
            sourceBuffers: {
                position_x: xPosition,
                position_y: yPosition,
            },
            feedbackBuffers: {
                filterValueIndex: filterValueIndexBuffer
            },
            elementCount: count
        });
        this.filterTransform.run({
            moduleSettings: {
                boundingBox: this.boundingBox,
                texture: this.polygonTexture
            }
        });
    }
}
