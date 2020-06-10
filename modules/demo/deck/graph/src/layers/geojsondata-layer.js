/* eslint-disable */

import {Layer, CompositeLayer, log, project32, picking} from '@deck.gl/core';
import {ScatterplotLayer, PathLayer, SolidPolygonLayer} from '@deck.gl/layers';
// Use primitive layer to avoid "Composite Composite" layers for now
import {replaceInRange} from './utils';
import GL from '@luma.gl/constants';
import { Buffer, Model } from '@luma.gl/core';
import { createIndexBufferTransform, computeIndexBuffers } from './index-buffers';
import {getGeojsonFeatures, separateGeojsonFeatures} from './geojsondata';

export class GeoJsonDataLayer extends CompositeLayer {
    static get layerName() { return 'GeoJsonDataLayer'; }
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
            const { length = 0, x, y, age, sex, education, income, cow  } = props.chunks.shift() || {};
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
                            elevation: new Buffer(gl, { data: cow, accessor: { ...elevationAccessor} }),
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
            return new GeoJsonDataChunkLayer(this.getSubLayerProps({
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

const defaultLineColor = [0, 0, 0, 255];
const defaultFillColor = [0, 0, 0, 255];

const xPositionsAccessor = { type: GL.FLOAT, fp64: true };
const yPositionsAccessor = { type: GL.FLOAT, fp64: true };
const radiusAccessor = { type: GL.UNSIGNED_BYTE, integer: false };
const colorsAccessor = { type: GL.UNSIGNED_BYTE, integer: false };
const elevationAccessor = { type: GL.UNSIGNED_BYTE, integer: false };
const elementIndicesAccessor = { type: GL.UNSIGNED_INT, isIndexed: true };

const defaultProps = {
  stroked: true,
  filled: true,
  extruded: false,
  wireframe: false,

  lineWidthUnits: 'meters',
  lineWidthScale: 1,
  lineWidthMinPixels: 0,
  lineWidthMaxPixels: Number.MAX_SAFE_INTEGER,
  lineJointRounded: false,
  lineMiterLimit: 4,

  elevationScale: 1,

  pointRadiusScale: 1,
  pointRadiusMinPixels: 0, //  min point radius in pixels
  pointRadiusMaxPixels: Number.MAX_SAFE_INTEGER, // max point radius in pixels

  // Line and polygon outline color
  getLineColor: {type: 'accessor', value: defaultLineColor},
  // Point and polygon fill color
  getFillColor: {type: 'accessor', value: defaultFillColor},
  // Point radius
  getRadius: {type: 'accessor', value: 1},
  // Line and polygon outline accessors
  getLineWidth: {type: 'accessor', value: 1},
  // Polygon extrusion accessor
  getElevation: {type: 'accessor', value: 1000},
  // Optional material for 'lighting' shader module
  material: true
};

function getCoordinates(f) {
  return f.geometry.coordinates;
}

class GeoJsonDataChunkLayer extends Layer {
    initializeState() {
        this.state = {
          features: {}
        };
        if (this.props.getLineDashArray) {
          log.removed('getLineDashArray', 'PathStyleExtension')();
        }
      }
    
      updateState({props, changeFlags}) {
        if (!changeFlags.dataChanged) {
          return;
        }
        const features = getGeojsonFeatures(props.data);
        const wrapFeature = this.getSubLayerRow.bind(this);
    
        if (Array.isArray(changeFlags.dataChanged)) {
          const oldFeatures = this.state.features;
          const newFeatures = {};
          const featuresDiff = {};
          for (const key in oldFeatures) {
            newFeatures[key] = oldFeatures[key].slice();
            featuresDiff[key] = [];
          }
    
          for (const dataRange of changeFlags.dataChanged) {
            const partialFeatures = separateGeojsonFeatures(features, wrapFeature, dataRange);
            for (const key in oldFeatures) {
              featuresDiff[key].push(
                replaceInRange({
                  data: newFeatures[key],
                  getIndex: f => f.__source.index,
                  dataRange,
                  replace: partialFeatures[key]
                })
              );
            }
          }
          this.setState({features: newFeatures, featuresDiff});
        } else {
          this.setState({
            features: separateGeojsonFeatures(features, wrapFeature),
            featuresDiff: {}
          });
        }
      }
    
      /* eslint-disable complexity */
      renderLayers() {
        const {features, featuresDiff} = this.state;
        const {pointFeatures, lineFeatures, polygonFeatures, polygonOutlineFeatures} = features;
    
        // Layer composition props
        const {stroked, filled, extruded, wireframe, material, transitions} = this.props;
    
        // Rendering props underlying layer
        const {
          lineWidthUnits,
          lineWidthScale,
          lineWidthMinPixels,
          lineWidthMaxPixels,
          lineJointRounded,
          lineMiterLimit,
          pointRadiusScale,
          pointRadiusMinPixels,
          pointRadiusMaxPixels,
          elevationScale,
          lineDashJustified
        } = this.props;
    
        // Accessor props for underlying layers
        const {
          getLineColor,
          getFillColor,
          getRadius,
          getLineWidth,
          getLineDashArray,
          getElevation,
          updateTriggers
        } = this.props;
    
        const PolygonFillLayer = this.getSubLayerClass('polygons-fill', SolidPolygonLayer);
        const PolygonStrokeLayer = this.getSubLayerClass('polygons-stroke', PathLayer);
        const LineStringsLayer = this.getSubLayerClass('line-strings', PathLayer);
        const PointsLayer = this.getSubLayerClass('points', ScatterplotLayer);
    
        // Filled Polygon Layer
        const polygonFillLayer =
          this.shouldRenderSubLayer('polygons-fill', polygonFeatures) &&
          new PolygonFillLayer(
            {
              _dataDiff: featuresDiff.polygonFeatures && (() => featuresDiff.polygonFeatures),
    
              extruded,
              elevationScale,
              filled,
              wireframe,
              material,
              getElevation: this.getSubLayerAccessor(getElevation),
              getFillColor: this.getSubLayerAccessor(getFillColor),
              getLineColor: this.getSubLayerAccessor(getLineColor),
    
              transitions: transitions && {
                getPolygon: transitions.geometry,
                getElevation: transitions.getElevation,
                getFillColor: transitions.getFillColor,
                getLineColor: transitions.getLineColor
              }
            },
            this.getSubLayerProps({
              id: 'polygons-fill',
              updateTriggers: {
                getElevation: updateTriggers.getElevation,
                getFillColor: updateTriggers.getFillColor,
                getLineColor: updateTriggers.getLineColor
              }
            }),
            {
              data: polygonFeatures,
              getPolygon: getCoordinates
            }
          );
    
        const polygonLineLayer =
          !extruded &&
          stroked &&
          this.shouldRenderSubLayer('polygons-stroke', polygonOutlineFeatures) &&
          new PolygonStrokeLayer(
            {
              _dataDiff:
                featuresDiff.polygonOutlineFeatures && (() => featuresDiff.polygonOutlineFeatures),
    
              widthUnits: lineWidthUnits,
              widthScale: lineWidthScale,
              widthMinPixels: lineWidthMinPixels,
              widthMaxPixels: lineWidthMaxPixels,
              rounded: lineJointRounded,
              miterLimit: lineMiterLimit,
              dashJustified: lineDashJustified,
    
              getColor: this.getSubLayerAccessor(getLineColor),
              getWidth: this.getSubLayerAccessor(getLineWidth),
              getDashArray: this.getSubLayerAccessor(getLineDashArray),
    
              transitions: transitions && {
                getPath: transitions.geometry,
                getColor: transitions.getLineColor,
                getWidth: transitions.getLineWidth
              }
            },
            this.getSubLayerProps({
              id: 'polygons-stroke',
              updateTriggers: {
                getColor: updateTriggers.getLineColor,
                getWidth: updateTriggers.getLineWidth,
                getDashArray: updateTriggers.getLineDashArray
              }
            }),
            {
              data: polygonOutlineFeatures,
              getPath: getCoordinates
            }
          );
    
        const pathLayer =
          this.shouldRenderSubLayer('linestrings', lineFeatures) &&
          new LineStringsLayer(
            {
              _dataDiff: featuresDiff.lineFeatures && (() => featuresDiff.lineFeatures),
    
              widthUnits: lineWidthUnits,
              widthScale: lineWidthScale,
              widthMinPixels: lineWidthMinPixels,
              widthMaxPixels: lineWidthMaxPixels,
              rounded: lineJointRounded,
              miterLimit: lineMiterLimit,
              dashJustified: lineDashJustified,
    
              getColor: this.getSubLayerAccessor(getLineColor),
              getWidth: this.getSubLayerAccessor(getLineWidth),
              getDashArray: this.getSubLayerAccessor(getLineDashArray),
    
              transitions: transitions && {
                getPath: transitions.geometry,
                getColor: transitions.getLineColor,
                getWidth: transitions.getLineWidth
              }
            },
            this.getSubLayerProps({
              id: 'line-strings',
              updateTriggers: {
                getColor: updateTriggers.getLineColor,
                getWidth: updateTriggers.getLineWidth,
                getDashArray: updateTriggers.getLineDashArray
              }
            }),
            {
              data: lineFeatures,
              getPath: getCoordinates
            }
          );
    
        const pointLayer =
          this.shouldRenderSubLayer('points', pointFeatures) &&
          new PointsLayer(
            {
              _dataDiff: featuresDiff.pointFeatures && (() => featuresDiff.pointFeatures),
    
              stroked,
              filled,
              radiusScale: pointRadiusScale,
              radiusMinPixels: pointRadiusMinPixels,
              radiusMaxPixels: pointRadiusMaxPixels,
              lineWidthUnits,
              lineWidthScale,
              lineWidthMinPixels,
              lineWidthMaxPixels,
    
              getFillColor: this.getSubLayerAccessor(getFillColor),
              getLineColor: this.getSubLayerAccessor(getLineColor),
              getRadius: this.getSubLayerAccessor(getRadius),
              getLineWidth: this.getSubLayerAccessor(getLineWidth),
    
              transitions: transitions && {
                getPosition: transitions.geometry,
                getFillColor: transitions.getFillColor,
                getLineColor: transitions.getLineColor,
                getRadius: transitions.getRadius,
                getLineWidth: transitions.getLineWidth
              }
            },
            this.getSubLayerProps({
              id: 'points',
              updateTriggers: {
                getFillColor: updateTriggers.getFillColor,
                getLineColor: updateTriggers.getLineColor,
                getRadius: updateTriggers.getRadius,
                getLineWidth: updateTriggers.getLineWidth
              }
            }),
            {
              data: pointFeatures,
              getPosition: getCoordinates,
              highlightedObjectIndex: this._getHighlightedIndex(pointFeatures)
            }
          );
    
        return [
          // If not extruded: flat fill layer is drawn below outlines
          !extruded && polygonFillLayer,
          polygonLineLayer,
          pathLayer,
          pointLayer,
          // If extruded: draw fill layer last for correct blending behavior
          extruded && polygonFillLayer
        ];
      }
      /* eslint-enable complexity */
    
      _getHighlightedIndex(data) {
        const {highlightedObjectIndex} = this.props;
        return Number.isFinite(highlightedObjectIndex)
          ? data.findIndex(d => d.__source.index === highlightedObjectIndex)
          : null;
      }
}

GeoJsonDataChunkLayer.layerName = 'GeoJsonDataChunkLayer';
GeoJsonDataChunkLayer.defaultProps = defaultProps;
