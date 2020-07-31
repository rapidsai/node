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

    controlPoint = vec3(0., 0., 1.);
    sourcePosition = vec3(0., 0., 1.);
    targetPosition = vec3(0., 0., 1.);

    if (edge.x < numNodesLoaded) {
        ivec2 xIdx = getTexCoord(edge.x);
        sourcePosition = vec3(
            texelFetch(nodeXPositions, xIdx, 0).x,
            texelFetch(nodeYPositions, xIdx, 0).x,
            0.
        );
    }

    if (edge.y < numNodesLoaded) {
        ivec2 yIdx = getTexCoord(edge.y);
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
        float maxBundleSize = length(targetPosition - sourcePosition) * 0.15;
        float eindex = float(bundle.x) + mod(float(bundle.x), 2.);
        float bcount = float(bundle.y);
        float direction = mix(1., -1., mod(bcount, 2.));

        // If all the edges in the bundle fit into maxBundleSize,
        // separate the edges without overlap via 'stroke * eindex'.
        // Otherwise allow edges to overlap.
        float size = mix(
            (strokeWidth * 2. * eindex),
            (maxBundleSize / strokeWidth) * (eindex / bcount),
            step(maxBundleSize, bcount * strokeWidth * 2.)
        ) + maxBundleSize;

        controlPoint = vec3((midp + (unit * size * direction)).xy, 0.);
    }
}
`;
