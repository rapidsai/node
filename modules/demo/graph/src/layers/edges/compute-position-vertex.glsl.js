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

export default `\
#version 300 es
#define SHADER_NAME compute-position-vertex

precision highp float;
precision highp sampler2D;

uniform float strokeWidth;
uniform uint textureWidth;
uniform uint numNodesLoaded;
uniform sampler2D nodeXPositions;
uniform sampler2D nodeYPositions;

in uvec2 edge;
in uvec2 bundle;

out vec3 controlPoint;
out vec3 sourcePosition;
out vec3 targetPosition;

ivec2 getTexCoord(uint id) {
    uint y = id / textureWidth;
    uint x = id - textureWidth * y;
    return ivec2(x, y);
}

void main(void) {

    uint source = min(edge.x, edge.y);
    uint target = max(edge.x, edge.y);

    controlPoint = vec3(0., 0., 1.);
    sourcePosition = vec3(0., 0., 1.);
    targetPosition = vec3(0., 0., 1.);

    if (source < numNodesLoaded) {
        ivec2 xIdx = getTexCoord(source);
        sourcePosition = vec3(
            texelFetch(nodeXPositions, xIdx, 0).x,
            texelFetch(nodeYPositions, xIdx, 0).x,
            0.
        );
    }

    if (target < numNodesLoaded) {
        ivec2 yIdx = getTexCoord(target);
        targetPosition = vec3(
            texelFetch(nodeXPositions, yIdx, 0).x,
            texelFetch(nodeYPositions, yIdx, 0).x,
            0.
        );
    }

    // Compute the quadratic bezier control point for this edge
    if ((sourcePosition.z + targetPosition.z) < 1.0) {

        vec3 diff = normalize(targetPosition - sourcePosition);
        vec3 midp = (targetPosition + sourcePosition) * .5;
        vec3 unit = vec3(-diff.y, diff.x, 1.);

        float stroke = strokeWidth;
        float maxBundleSize = length(targetPosition - sourcePosition) * 0.25;

        float eindex = float(bundle.x);
        float bcount = float(bundle.y);

        float direction = 1.0; // all edges will bend in the same direction

        float size = mix(maxBundleSize * .5, maxBundleSize * 1.5, eindex / bcount);

        controlPoint = vec3((midp + (unit * size * direction)).xy, 0.);
    }
}
`;
