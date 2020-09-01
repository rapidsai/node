import Url from 'url';
import * as zmq from 'zeromq';
import { Deck, OrthographicView, COORDINATE_SYSTEM, log as deckLog } from '@deck.gl/core';
import ArrowGraphLayer from './layers/arrow-graph-layer';
import { CUDAMemory, CUDA } from '@nvidia/cuda';

let numNodes = 0;
let numEdges = 0;
let drawEdges = null;
let graphVersion = 0;
let redrawTimeout = null;
let onAfterRenderPromise = null;

const deck = new Deck({
    width: '100%',
    height: '100%',
    controller: true,
    initialViewState: {target: [0, 0, 0], zoom: -4},
    views: new OrthographicView({controller: true}),
    onViewStateChange: ({ viewState }) => redraw({viewState, drawEdges})
});

window._inputEventTarget = deck.canvas;

// const useTestData = true;
const useTestData = false;

(useTestData ? localRenderLoop() : remoteRenderLoop())
    .catch((e) => console.error('Main loop error:', e) || process.exit(1));

async function localRenderLoop() {
    await Promise.resolve().then(() => {
        const r = 5;
        const d = r * 2;
        const xOff = 100;
        const yOff = 100;
        const c = [0xFFFFFFFF, 0xFFFF0000, 0xFF00FF00, 0xFF0000FF];

        const edgeUpdates = [(() => {
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
                edge: Uint32Array.from(edges.flatMap((e) => e)),
                bundle: Uint32Array.from(bundles.flatMap((b) => b)),
                color: Uint32Array.from(edges.flatMap((e) => e).map((n) => c[n]))
            };
        })()];

        const nodeUpdates = [(() => {
            return {
                offset: 0,
                x: Float32Array.from([
                    (xOff * +0)    , // center (white)
                    (xOff * +1) + d, // bottom right (blue)
                    (xOff * -1) - d, // bottom left (green)
                    (xOff * +0)    , // top middle (red)
                ]),
                y: Float32Array.from([
                    (yOff * +0) + d, // center (white)
                    (yOff * +1) - d, // bottom right (blue)
                    (yOff * +1) - d, // bottom left (green)
                    (yOff * -1) - d, // top middle (red)
                ]),
                size: Uint8Array.from([r, r, r, r]),
                color: Uint32Array.from(c),
            }
        })()];

        numEdges = edgeUpdates[0].length = edgeUpdates[0].edge.length / 2;
        numNodes = nodeUpdates[0].length = nodeUpdates[0].x.length;
        console.log('node_x:',      [...nodeUpdates[0].x]);
        console.log('node_y:',      [...nodeUpdates[0].y]);
        console.log('node_size:',   [...nodeUpdates[0].size]);
        console.log('node_color:',  [...nodeUpdates[0].color]);
        console.log('edge_list:',   [...edgeUpdates[0].edge]);
        console.log('edge_bundle:', [...edgeUpdates[0].bundle]);
        console.log('edge_color:',  [...edgeUpdates[0].color]);

        // console.log({
        //     edges: edgeUpdates[0],
        //     nodes: nodeUpdates[0],
        // });
        return redraw({
            nodeUpdates,
            edgeUpdates,
            drawEdges: drawEdges = true,
            graphVersion: ++graphVersion
        });
    });
}


async function remoteRenderLoop() {
    const mapToObject = (m) => [...m.entries()].reduce((xs, [k, v]) => ({...xs, [k]: v}), {});
    const loadIpcHandle = (handle) => new CUDAMemory(CUDA.ipc.openMemHandle(Buffer.from(handle)));
    const loadIpcHandles = (handles, names) => handles.filter(Boolean)
        .filter(({name, data}) => (data || []).length > 0 && ~names.indexOf(name))
        // .filter(({name, data}) => (data || []).length > 0)
        .reduce((xs, {name, data}) => xs.set(name, loadIpcHandle(data)), new Map());

    const closeMemHandle = ({buffer}) => { try { CUDA.ipc.closeMemHandle(buffer); } catch (e) {} };

    const ipchs = new zmq.Pull();
    const ready = new zmq.Request();

    const { protocol, hostname, port } = Url.parse(process.argv[2]);
    const ipchsUrl = `${protocol}//${hostname}:${port}`;
    const readyUrl = `${protocol}//${hostname}:${+port + 1}`;

    ipchs.connect(ipchsUrl);
    ready.connect(readyUrl);

    await ready.send('ready');
    await ready.receive();

    for await (const msg of ipchs) {
        // console.log(msg);
        let { node: nodeTokens, edge: edgeTokens } = JSON.parse(msg);
        // console.log(nodeTokens);
        nodeTokens = (nodeTokens || []).filter(Boolean);
        edgeTokens = (edgeTokens || []).filter(Boolean);
        // console.log(nodeTokens);
        let nodeIPCBuffers = loadIpcHandles(nodeTokens, ['x', 'y', 'size', 'color']);
        let edgeIPCBuffers = loadIpcHandles(edgeTokens, ['edge', 'color', 'bundle']);
        if (nodeIPCBuffers.size + edgeIPCBuffers.size > 0) {
            // console.log(nodeIPCBuffers);
            numNodes = nodeIPCBuffers.has('color') ? nodeIPCBuffers.get('color').byteLength / 4 : numNodes;
            numEdges = edgeIPCBuffers.has('color') ? edgeIPCBuffers.get('color').byteLength / 8 : numEdges;
            const nodeUpdates = nodeIPCBuffers.size === 0 ? [] : [{ length: numNodes, offset: 0, ...mapToObject(nodeIPCBuffers) }];
            // nodeUpdates[0] && console.log(nodeUpdates[0]);
            const edgeUpdates = edgeIPCBuffers.size === 0 ? [] : [{ length: numEdges, offset: 0, ...mapToObject(edgeIPCBuffers) }];
            // edgeUpdates[0] && console.log(edgeUpdates[0]);
            await redraw({nodeUpdates, edgeUpdates, drawEdges: drawEdges = null, graphVersion: ++graphVersion});
            nodeIPCBuffers.forEach(closeMemHandle);
            edgeIPCBuffers.forEach(closeMemHandle);
        }
        await ready.send('ready');
        if (`${await ready.receive()}` === 'close') {
            break;
        }
    }

    console.log('done');

    ipchs.close();
    ready.close();

    await redraw({drawEdges: drawEdges = true});
}

function redraw({ nodeUpdates = [], edgeUpdates = [], ...rest }) {
    const nextProps = {
        width: window.clientWidth,
        height: window.clientHeight,
        layers: [
            new ArrowGraphLayer(
                {
                    id: 'graph',
                    nodeUpdates,
                    edgeUpdates,
                    numNodes: numNodes,
                    numEdges: numEdges,
                    version: graphVersion,
                    edgeWidth: useTestData ? 5 : 10,
                    edgeOpacity: useTestData ? 0.5 : 0.5,
                    coordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
                },
                { drawEdges }
            )
        ]
    };

    rest.viewState !== undefined && (nextProps.viewState = rest.viewState);

    if (!onAfterRenderPromise) {
        onAfterRenderPromise = new Promise((resolve) => {
            nextProps.onAfterRender = () => {
                onAfterRenderPromise = null;
                resolve();
            }
        });
    }
    deck.setProps(nextProps);

    if (drawEdges === false) {
        redrawTimeout !== null && clearTimeout(redrawTimeout);
        redrawTimeout = setTimeout(() => {
            redrawTimeout = null;
            redraw({drawEdges: drawEdges = true});
        }, 350);
    }

    return onAfterRenderPromise;
}
