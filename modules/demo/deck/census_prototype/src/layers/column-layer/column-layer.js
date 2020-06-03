/* eslint-disable */

import {Layer, project32, gouraudLighting, picking, CompositeLayer } from '@deck.gl/core';
import GL from '@luma.gl/constants';
import {Model} from '@luma.gl/core';
import ColumnGeometry from './column-geometry';

export class ColumnLayer extends CompositeLayer {
  static get layerName() { return 'ColumnLayer'; }
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
                  id: `node-layer-${chunks.length}`,
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

import vs from './column-layer-vertex.glsl';
import fs from './column-layer-fragment.glsl';

// const DEFAULT_COLOR = [0, 0, 0, 255];

const defaultProps = {
  diskResolution: {type: 'number', min: 4, value: 20},
  vertices: null,
  radius: {type: 'number', min: 0, value: 1000},
  angle: {type: 'number', value: 0},
  offset: {type: 'array', value: [0, 0]},
  coverage: {type: 'number', min: 0, max: 1, value: 1},
  elevationScale: {type: 'number', min: 0, value: 1},

  lineWidthUnits: 'meters',
  lineWidthScale: 1,
  lineWidthMinPixels: 0,
  lineWidthMaxPixels: Number.MAX_SAFE_INTEGER,

  extruded: true,
  wireframe: false,
  filled: true,
  stroked: false,

  // getPosition: {type: 'accessor', value: x => x.position},
  // getFillColor: {type: 'accessor', value: DEFAULT_COLOR},
  // getLineColor: {type: 'accessor', value: DEFAULT_COLOR},
  // getLineWidth: {type: 'accessor', value: 1},
  // getElevation: {type: 'accessor', value: 1000},
  // material: true,
  // getColor: {deprecatedFor: ['getFillColor', 'getLineColor']}
};

class ColumnChunkLayer extends Layer {
  getShaders(id) {
    return super.getShaders({vs, fs, modules: [project32, gouraudLighting, picking]});
  }

  initializeState() {
    const attributeManager = this.getAttributeManager();
    /* eslint-disable max-len */
    attributeManager.addInstanced({
      instancePositions: {
        size: 3,
        type: GL.DOUBLE,
        fp64: this.use64bitPositions(),
        transition: true,
        accessor: 'getPosition'
      },
      instanceElevations: {
        size: 1,
        transition: true,
        accessor: 'getElevation'
      },
      instanceFillColors: {
        size: this.props.colorFormat.length,
        type: GL.UNSIGNED_BYTE,
        normalized: true,
        transition: true,
        accessor: 'getFillColor',
        defaultValue: DEFAULT_COLOR
      },
      instanceLineColors: {
        size: this.props.colorFormat.length,
        type: GL.UNSIGNED_BYTE,
        normalized: true,
        transition: true,
        accessor: 'getLineColor',
        defaultValue: DEFAULT_COLOR
      },
      instanceStrokeWidths: {
        size: 1,
        accessor: 'getLineWidth',
        transition: true
      }
    });
    /* eslint-enable max-len */
  }

  updateState({props, oldProps, changeFlags}) {
    super.updateState({props, oldProps, changeFlags});

    const regenerateModels = changeFlags.extensionsChanged;

    if (regenerateModels) {
      const {gl} = this.context;
      if (this.state.model) {
        this.state.model.delete();
      }
      this.setState({model: this._getModel(gl)});
      this.getAttributeManager().invalidateAll();
    }

    if (
      regenerateModels ||
      props.diskResolution !== oldProps.diskResolution ||
      props.vertices !== oldProps.vertices
    ) {
      this._updateGeometry(props);
    }
  }

  getGeometry(diskResolution, vertices) {
    const geometry = new ColumnGeometry({
      radius: 1,
      height: 2,
      vertices,
      nradial: diskResolution
    });

    let meanVertexDistance = 0;
    if (vertices) {
      for (let i = 0; i < diskResolution; i++) {
        const p = vertices[i];
        const d = Math.sqrt(p[0] * p[0] + p[1] * p[1]);
        meanVertexDistance += d / diskResolution;
      }
    } else {
      meanVertexDistance = 1;
    }
    this.setState({
      edgeDistance: Math.cos(Math.PI / diskResolution) * meanVertexDistance
    });

    return geometry;
  }

  _getModel(gl) {
    return new Model(
      gl,
      Object.assign({}, this.getShaders(), {
        id: this.props.id,
        isInstanced: true
      })
    );
  }

  _updateGeometry({diskResolution, vertices}) {
    const geometry = this.getGeometry(diskResolution, vertices);

    this.setState({
      fillVertexCount: geometry.ColumnChunkLayerattributes.POSITION.value.length / 3,
      wireframeVertexCount: geometry.indices.value.length
    });

    this.state.model.setProps({geometry});
  }

  draw({uniforms}) {
    const {viewport} = this.context;
    const {
      lineWidthUnits,
      lineWidthScale,
      lineWidthMinPixels,
      lineWidthMaxPixels,

      elevationScale,
      extruded,
      filled,
      stroked,
      wireframe,
      offset,
      coverage,
      radius,
      angle
    } = this.props;
    const {model, fillVertexCount, wireframeVertexCount, edgeDistance} = this.state;

    const widthMultiplier = lineWidthUnits === 'pixels' ? viewport.metersPerPixel : 1;

    model.setUniforms(
      Object.assign({}, uniforms, {
        radius,
        angle: (angle / 180) * Math.PI,
        offset,
        extruded,
        coverage,
        elevationScale,
        edgeDistance,
        widthScale: lineWidthScale * widthMultiplier,
        widthMinPixels: lineWidthMinPixels,
        widthMaxPixels: lineWidthMaxPixels
      })
    );

    // When drawing 3d: draw wireframe first so it doesn't get occluded by depth test
    if (extruded && wireframe) {
      model.setProps({isIndexed: true});
      model
        .setVertexCount(wireframeVertexCount)
        .setDrawMode(GL.LINES)
        .setUniforms({isStroke: true})
        .draw();
    }
    if (filled) {
      model.setProps({isIndexed: false});
      model
        .setVertexCount(fillVertexCount)
        .setDrawMode(GL.TRIANGLE_STRIP)
        .setUniforms({isStroke: false})
        .draw();
    }
    // When drawing 2d: draw fill before stroke so that the outline is always on top
    if (!extruded && stroked) {
      model.setProps({isIndexed: false});
      // The width of the stroke is achieved by flattening the side of the cylinder.
      // Skip the last 1/3 of the vertices which is the top.
      model
        .setVertexCount((fillVertexCount * 2) / 3)
        .setDrawMode(GL.TRIANGLE_STRIP)
        .setUniforms({isStroke: true})
        .draw();
    }
  }
}

ColumnChunkLayer.layerName = 'ColumnChunkLayer';
ColumnChunkLayer.defaultProps = defaultProps;
