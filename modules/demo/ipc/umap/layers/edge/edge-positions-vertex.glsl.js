// Copyright (c) 2015 - 2017 Uber Technologies, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

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

export default `\
#version 410

#define MAX_BUNDLE_SIZE 100.0

in uvec2 edge;
in uvec2 bundle;

uniform sampler2D nodePositions;
uniform uint loadedNodeCount;
uniform float strokeWidth;
uniform uint width;

out vec3 controlPoints;
out vec3 sourcePositions;
out vec3 targetPositions;

ivec2 getTexCoord(uint id) {
    uint y = id / width;
    uint x = id - width * y;
    return ivec2(x, y);
}

void main(void) {

    controlPoints = vec3(0., 0., 1.);
    sourcePositions = vec3(0., 0., 1.);
    targetPositions = vec3(0., 0., 1.);

    if (edge.x < loadedNodeCount) {
        sourcePositions = vec3(texelFetch(nodePositions, getTexCoord(edge.x), 0).xy, 0.);
    }
    if (edge.y < loadedNodeCount) {
        targetPositions = vec3(texelFetch(nodePositions, getTexCoord(edge.y), 0).xy, 0.);
    }

    // Compute the quadratic bezier control point for this edge
    if ((sourcePositions.z + targetPositions.z) < 1.0) {

        uint uindex = bundle.x;
        float bundleSize = bundle.y;
        float stroke = max(strokeWidth, 1.);
        float eindex = uindex + mod(uindex, 2);
        int direction = int(mix(1, -1, mod(uindex, 2)));

        // If all the edges in the bundle fit into MAX_BUNDLE_SIZE,
        // separate the edges without overlap via 'stroke * eindex'.
        // Otherwise allow edges to overlap.
        float size = mix(
            stroke * eindex,
            (MAX_BUNDLE_SIZE * .5 / stroke)
                * (eindex / bundleSize),
            step(MAX_BUNDLE_SIZE, bundleSize * strokeWidth)
        );

        vec3 midp = (sourcePositions + targetPositions) * 0.5;
        vec3 dist = (targetPositions - sourcePositions);
        vec3 unit = normalize(dist) * vec3(-1., 1., 1.);

        // handle horizontal and vertical edges
        if (unit.x == 0. || unit.y == 0.) {
            unit = mix(
                vec3(1., 0., unit.z), // if x is 0, x=1, y=0
                vec3(0., 1., unit.z), // if y is 0, x=0, y=1
                step(0.5, unit.x)
            );
        }

        controlPoints = vec3((midp + (unit * size * direction)).xy, 0.);
    }
}
`;
