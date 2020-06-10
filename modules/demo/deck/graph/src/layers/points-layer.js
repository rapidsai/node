/* eslint-disable */

import GL from '@luma.gl/constants';
import { Buffer, Model } from '@luma.gl/core';
import { Layer, CompositeLayer, project32, picking } from '@deck.gl/core';
import { createIndexBufferTransform, computeIndexBuffers } from './index-buffers';

export class PointsLayer extends CompositeLayer {
    static get layerName() { return 'PointsLayer'; }
    initializeState() {
        this.setState({
            chunks: [],
            zoom: undefined,
            computeStridedIndexBuffer: createIndexBufferTransform(this.context.gl)
        });
    }
    shouldUpdateState({ props, oldProps, changeFlags, ...rest }) {
        if (props.chunks && props.chunks.length) {
            return props.chunks.some(({ length }) => length > 0);
        }
        // Return true so we resample element indices on pan/zoom
        if (changeFlags.viewportChanged && this.state.chunks.length > 0) {
            return true;
        }
        return super.shouldUpdateState({ props, changeFlags, ...rest });
    }
    updateState({ props, context, changeFlags }) {

        const { gl } = context;
        const { chunks, zoom } = this.state;
        const zoomChanged = changeFlags.viewportChanged && (zoom !== context.viewport.zoom);

        let newChunk = false;

        while ((props.chunks || []).length > 0) {
            const { length = 0, x, y, age, sex, education, income, cow } = props.chunks.shift() || {};
            if (length > 0) {
                newChunk = true;
                chunks.push({
                    // interaction
                    pickable: true,
                    autoHighlight: true,
                    highlightColor: [255, 255, 255, 100],
                    // render info
                    opacity: 1,
                    radiusScale: this.props.radiusScale || 1,
                    radiusMinPixels: this.props.radiusMinPixels || 0,
                    radiusMaxPixels: this.props.radiusMaxPixels || 50,
                    // layer info
                    vertexCount: length,
                    // numElements: length,
                    id: `points-layer-${chunks.length}`,
                    // 
                    // Sub-sample which elements to render based on the zoom level.
                    // 
                    // Most points are too small to see when the camera is zoomed "out."
                    // Fragment culling is fairly robust on a few million points. Above that,
                    // even running the vertex shader so many times is prohibitively expensive.
                    //
                    // So instead we compute a subset of strided element indices. For example,
                    // if zoom is ~1, we'll compute a buffer of 1% of the total indices.
                    // As the zoom approaches 20, our sample size will approach 100%.
                    ...computeIndexBuffers(gl, length, this.state.computeStridedIndexBuffer),
                    data: {
                        attributes: {
                            radius: new Buffer(gl, { data: age, accessor: { ...radiusAccessor } }),
                            fillColors: new Buffer(gl, { data: sex, accessor: { ...colorsAccessor } }),
                            xPositions: new Buffer(gl, { data: x, accessor: { ...xPositionsAccessor } }),
                            yPositions: new Buffer(gl, { data: y, accessor: { ...yPositionsAccessor } }),
                        }
                    },
                });
            }
        }

        if ((newChunk || zoomChanged) && chunks.length > 0) {
            this.setState({
                zoom: context.viewport.zoom,
                chunks: this.state.chunks.slice()
            });
        }
    }
    renderLayers() {
        const zoom = Math.max(this.context.viewport.zoom, 1) / 15;
        const sample = 1 -  Math.sqrt(1 - Math.pow(Math.min(zoom, 1), 2));
        return this.state.chunks.map((chunk, i) => {
            const { vertexCount, normalIndex, randomIndex } = chunk;
            const elements = zoom >= 10 ? normalIndex : randomIndex;
            const numElements = zoom >= 10 ? vertexCount : Math.round(sample * vertexCount);
            // console.log({
            //     i,
            //     numElements,
            //     elementsLen: elements.byteLength / elements.accessor.BYTES_PER_VERTEX,
            //     zoom: +zoom.toPrecision(2),
            //     sample: +sample.toPrecision(2),
            // });
            return new PointsChunkLayer(this.getSubLayerProps({
                ...chunk,
                vertexCount: numElements,
                data: {
                    ...chunk.data,
                    attributes: {
                        ...chunk.data.attributes,
                        lineColors: { ...chunk.data.attributes.fillColors },
                        xPositions64Low: { ...chunk.data.attributes.xPositions },
                        yPositions64Low: { ...chunk.data.attributes.yPositions },
                        elementIndices: { ...elementIndicesAccessor, buffer: elements },
                    }
                }
            }));
        });
    }
}

import vs from './point-layer-vertex.glsl';
import fs from './point-layer-fragment.glsl';

const xPositionsAccessor = { type: GL.FLOAT, fp64: true };
const yPositionsAccessor = { type: GL.FLOAT, fp64: true };
const radiusAccessor = { type: GL.UNSIGNED_BYTE, integer: false };
const colorsAccessor = { type: GL.UNSIGNED_BYTE, integer: false };
const elementIndicesAccessor = { type: GL.UNSIGNED_INT, isIndexed: true };

const defaultProps = {
    filled: { type: 'boolean', value: true },
    stroked: { type: 'boolean', value: true },
    opacity: { type: 'number', min: 0, value: 1 },
    radiusScale: { type: 'number', min: 0, value: 1 },
    radiusMinPixels: { type: 'number', min: 0, value: 0 }, //  min point radius in pixels
    radiusMaxPixels: { type: 'number', min: 0, value: Number.MAX_SAFE_INTEGER }, // max point radius in pixels
};

class PointsChunkLayer extends Layer {
    getShaders(id) {
        return super.getShaders({ vs, fs, modules: [project32, picking] });
    }
    initializeState() {
        this.getAttributeManager().remove('instancePickingColors');
        this.getAttributeManager().add({
            radius: { ...radiusAccessor, size: 1, accessor: 'getRadius' },
            fillColors: { ...colorsAccessor, size: 1, accessor: 'getFillColors' },
            xPositions: { ...xPositionsAccessor, size: 1, accessor: 'getXPositions' },
            yPositions: { ...yPositionsAccessor, size: 1, accessor: 'getYPositions' },
            elementIndices: { ...elementIndicesAccessor, size: 1, accessor: 'getElementIndices' },
        });
    }
    updateState(opts = {}) {
        // console.log(opts);
        super.updateState(opts);
        const { gl } = this.context;
        const { changeFlags } = opts;
        if (changeFlags.extensionsChanged) {
            if (this.state.model) {
                this.state.model.delete();
            }
            this.setState({ model: this._getModel(gl) });
            this.getAttributeManager().invalidateAll();
        }
    }
    draw({ uniforms = {} }) {
        this.state.model
            .setVertexCount(Math.max(0, this.props.vertexCount))
            .setUniforms({
                ...uniforms,
                ...this.props,
                stroked: this.props.stroked ? 1 : 0
            }).draw();
    }
    _getModel(gl) {
        return new Model(
            gl,
            Object.assign(this.getShaders(), {
                id: this.props.id,
                drawMode: gl.POINTS,
                vertexCount: 1,
                isIndexed: true,
                isInstanced: false,
                indexType: gl.UNSIGNED_INT
            })
        );
    }
}

PointsChunkLayer.layerName = 'PointsChunkLayer';
PointsChunkLayer.defaultProps = defaultProps;
