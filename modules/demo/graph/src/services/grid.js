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
    const { bbox, edges, nodes } = testData(props);
    return {
        bbox,
        graph: {
            nodeRadiusScale: 1,
            data: { edges, nodes },
            numNodes: nodes.length,
            numEdges: edges.length,
        }
    };
}

function testData() {
    const numBatches = 2;
    const numElements = 16777215 / 3;
    const radius = 5, diameter = radius * 2;
    const numRows = Math.round(Math.sqrt(numElements * numBatches));
    const numCols = Math.round(Math.sqrt(numElements * numBatches));

    const nodeRadius = Uint8Array.from({ length: numRows * numCols }, (_) => radius);
    const nodeElementIndices = Uint32Array.from({ length: nodeRadius.length }, (_, i) => i);
    const nodeFillColors = new Uint8Array(nodeRadius.length * 4);
    const nodeXPositions = new Float32Array(nodeRadius.length);
    const nodeYPositions = new Float32Array(nodeRadius.length);

    const positionStep = diameter * 1.5;
    const positionPadding = diameter * 0.25;

    for (let i = -1; ++i < numRows;) {
        for (let j = -1; ++j < numCols;) {
            const n = (i * numRows) + j;
            nodeXPositions[n] = (i * positionStep * 2) - positionPadding;
            nodeYPositions[n] = (j * positionStep) - positionPadding;
            nodeFillColors[n * 4 + 0] = n % 255 * (n % 3 === 0);
            nodeFillColors[n * 4 + 1] = n % 255 * (n % 3 === 1);
            nodeFillColors[n * 4 + 2] = n % 255 * (n % 3 === 2);
            nodeFillColors[n * 4 + 3] = 255;
        }
    }

    const edgePairs = Array.from({ length: 500 }, (_, i) => {
        switch (i % 4) {
            case 0: return [0, nodeRadius.length - 1];
            case 1: return [nodeRadius.length - 1, 0];
            case 2: return [0, 1];
            case 3: return [80280, 0];
        }
    });

    return {
        bbox: [
            (positionStep * numRows) * -0.5,
            (positionStep * numRows) *  0.5,
            (positionStep * numCols) * -0.5,
            (positionStep * numCols) *  0.5,
        ],
        edges: {
            length: edgePairs.length,
            attributes: edgeAttributes(edgePairs, new Uint32Array(nodeFillColors.buffer))
        },
        nodes: {
            length: nodeRadius.length,
            attributes: {
                nodeRadius,
                nodeFillColors: new Uint32Array(nodeFillColors.buffer),
                nodeLineColors: new Uint32Array(nodeFillColors.buffer),
                nodeXPositions,
                nodeYPositions,
                nodeElementIndices,
            }
        }
    };
}

function edgeAttributes(edges, nodeColors) {
    const edgeColors = Uint32Array.from(edges.flatMap(([x, y]) => [nodeColors[x], nodeColors[y]]));
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
        edgeList: Uint32Array.from(edges.flat()),
        edgeBundles: Uint32Array.from(bundles.flat()),
        edgeColors,
    };
}