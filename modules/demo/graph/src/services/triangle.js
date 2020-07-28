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

export default async function loadGraphData(props = {}) {
    const { edges, nodes } = testData(props);
    return {
        graph: {
            nodeRadiusScale: 1,
            data: { edges, nodes },
            numNodes: nodes.length,
            numEdges: edges.length,
        }
    };
}

function testData(props = {}) {
    const radius = 5;
    const diameter = radius * 2;
    const xOffset = radius * 20;
    const yOffset = radius * 20;
    const colors = [0xFFFFFFFF, 0xFFFF0000, 0xFF00FF00, 0xFF0000FF];

    const edges = (() => {
        const edges = [
            [0, 1], [0, 2], [0, 3],
            [1, 0], [1, 2], [1, 3], [1, 3],
            [2, 0], [2, 1], [2, 3],
            [3, 1], [3, 1], [3, 2], [3, 0], [3, 0],
        ];
        const bundles = (() => {
            const { keys, offsets, lengths } = edges.reduce(({ keys, offsets, lengths }, e, i) => {
                const k = `${e.sort((a, b) => a - b)}`;
                lengths[k] = (lengths[k] || 0) + 1;
                offsets[i] = (lengths[k] - 1);
                keys[i] = k;
                return { keys, offsets, lengths };
            }, { keys: [], offsets: [], lengths: {} });
            return keys.map((k, i) => [offsets[i], lengths[k]]);
        })();
        return {
            offset: 0,
            length: edges.length,
            attributes: {
                edgeList: Uint32Array.from(edges.flatMap((e) => e)),
                edgeBundles: Uint32Array.from(bundles.flatMap((b) => b)),
                edgeColors: Uint32Array.from(edges.flatMap((e) => e).map((n) => colors[n]))
            },
        };
    })();

    const nodes = (() => {
        return {
            offset: 0,
            length: 4,
            attributes: {
                nodeRadius: Uint8Array.from([radius, radius, radius, radius]),
                nodeFillColors: Uint32Array.from(colors),
                nodeLineColors: Uint32Array.from(colors),
                nodeXPositions: Float32Array.from([
                    (xOffset * +0)           , // center (white)
                    (xOffset * +1) + diameter, // bottom right (blue)
                    // (xOffset * +0) + diameter, // bottom right (blue)
                    (xOffset * -1) - diameter, // bottom left (green)
                    (xOffset * +0)           , // top middle (red)
                ]),
                nodeYPositions: Float32Array.from([
                    (yOffset * +0) + diameter, // center (white)
                    (yOffset * +1) - diameter, // bottom right (blue)
                    // (yOffset * +0) + diameter, // bottom right (blue)
                    (yOffset * +1) - diameter, // bottom left (green)
                    (yOffset * -1) - diameter, // top middle (red)
                ]),
                nodeElementIndices: Uint32Array.from([0, 1, 2, 3]),
            },
        }
    })();

    // console.log('edgeList:',           [...edges.attributes.edgeList]);
    // console.log('edgeBundles:',        [...edges.attributes.edgeBundles]);
    // console.log('edgeColors:',         [...edges.attributes.edgeColors]);

    // console.log('nodeRadius:',         [...nodes.attributes.nodeRadius]);
    // console.log('nodeFillColors:',     [...nodes.attributes.nodeFillColors]);
    // console.log('nodeLineColors:',     [...nodes.attributes.nodeLineColors]);
    // console.log('nodeXPositions:',     [...nodes.attributes.nodeXPositions]);
    // console.log('nodeYPositions:',     [...nodes.attributes.nodeYPositions]);
    // console.log('nodeElementIndices:', [...nodes.attributes.nodeElementIndices]);

    return { edges, nodes };
}
