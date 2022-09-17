import { HexagonLayer } from '@deck.gl/aggregation-layers';
import { AmbientLight, LightingEffect, PointLight } from '@deck.gl/core';
import DeckGL from '@deck.gl/react';
import * as React from 'react';
import { Component } from 'react';
import { StaticMap } from 'react-map-gl';

// Set your mapbox token here
const MAPBOX_TOKEN =
  'pk.eyJ1Ijoid21qcGlsbG93IiwiYSI6ImNrN2JldzdpbDA2Ym0zZXFzZ3oydXN2ajIifQ.qPOZDsyYgMMUhxEKrvHzRA';  // eslint-disable-line

// Source data CSV
const DATA_URL =
  'https://raw.githubusercontent.com/uber-common/deck.gl-data/master/examples/3d-heatmap/heatmap-data.csv';  // eslint-disable-line

const ambientLight = new AmbientLight({ color: [255, 255, 255], intensity: 1.0 });

const pointLight1 =
  new PointLight({ color: [255, 255, 255], intensity: 0.8, position: [-0.144528, 49.739968, 80000] });

const pointLight2 =
  new PointLight({ color: [255, 255, 255], intensity: 0.8, position: [-3.807751, 54.104682, 8000] });

const lightingEffect = new LightingEffect({ ambientLight, pointLight1, pointLight2 });

const material = {
  ambient: 0.64,
  diffuse: 0.6,
  shininess: 32,
  specularColor: [51, 51, 51]
};

const INITIAL_VIEW_STATE = {
  longitude: -1.4157267858730052,
  latitude: 52.232395363869415,
  zoom: 6.6,
  minZoom: 5,
  maxZoom: 15,
  pitch: 40.5,
  bearing: -27.396674584323023
};

const colorRange =
  [[1, 152, 189], [73, 227, 206], [216, 254, 181], [254, 237, 177], [254, 173, 84], [209, 55, 78]];

const elevationScale = {
  min: 1,
  max: 50
};

/* eslint-disable react/no-deprecated */
export default class App extends Component {
  static get defaultColorRange() { return colorRange; }

  constructor(props) {
    super(props);
    this.state = { elevationScale: elevationScale.min };
    require('d3-request').csv(DATA_URL, (error, response) => {
      if (!error) { this.setState({ data: response.map(d => [Number(d.lng), Number(d.lat)]) }); }
    });
  }

  _renderLayers() {
    const { data } = this.state;
    const { radius = 1000, upperPercentile = 100, coverage = 1 } = this.props;

    return [new HexagonLayer({
      id: 'heatmap',
      colorRange,
      coverage,
      data,
      elevationRange: [0, 3000],
      elevationScale: data && data.length ? 50 : 0,
      extruded: true,
      getPosition: d => d,
      onHover: this.props.onHover,
      pickable: Boolean(this.props.onHover),
      radius,
      upperPercentile,
      material,

      transitions: { elevationScale: 3000 }
    })];
  }

  render() {
    const { mapStyle = 'https://basemaps.cartocdn.com/gl/dark-matter-nolabels-gl-style/style.json' } = this.props;

    return (
      <DeckGL
        layers={this._renderLayers()}
        effects={[lightingEffect]}
        initialViewState={INITIAL_VIEW_STATE}
        controller={true}>
        <StaticMap
          reuseMaps
          mapStyle={mapStyle}
          preventStyleDiffing={true}
          mapboxApiAccessToken={MAPBOX_TOKEN}
        />
      </DeckGL>
    );
  }
}
