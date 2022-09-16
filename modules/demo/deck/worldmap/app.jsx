import { _SunLight as SunLight, AmbientLight, LightingEffect } from '@deck.gl/core';
import { GeoJsonLayer, PolygonLayer } from '@deck.gl/layers';
import DeckGL from '@deck.gl/react';
import * as React from 'react';
import { Component } from 'react';
import { render } from 'react-dom';
import { StaticMap } from 'react-map-gl';

// Set your mapbox token here
const MAPBOX_TOKEN =
  'pk.eyJ1Ijoid21qcGlsbG93IiwiYSI6ImNrN2JldzdpbDA2Ym0zZXFzZ3oydXN2ajIifQ.qPOZDsyYgMMUhxEKrvHzRA';  // eslint-disable-line

// Source data GeoJSON
const DATA_URL = 'https://raw.githubusercontent.com/johan/world.geo.json/master/countries.geo.json';

export const INITIAL_VIEW_STATE = {
  latitude: 49.254,
  longitude: -123.13,
  zoom: 1,
  maxZoom: 16,
  pitch: 45,
  bearing: 0
};

const ambientLight = new AmbientLight({ color: [255, 255, 255], intensity: 1.0 });

const dirLight = new SunLight(
  { timestamp: Date.UTC(2019, 7, 1, 22), color: [255, 255, 255], intensity: 1.0, _shadow: true });

const landCover = [[[-123.0, 49.196], [-123.0, 49.324], [-123.306, 49.324], [-123.306, 49.196]]];

export default class App extends Component {
  constructor(props) {
    super(props);

    this.state = { hoveredObject: null };
    this._onHover = this._onHover.bind(this);
    this._renderTooltip = this._renderTooltip.bind(this);

    const lightingEffect = new LightingEffect({ ambientLight, dirLight });
    lightingEffect.shadowColor = [0, 0, 0, 1];
    this._effects = [lightingEffect];
  }

  _onHover({ x, y, object }) { this.setState({ x, y, hoveredObject: object }); }

  _renderLayers() {
    const { data = DATA_URL } = this.props;

    return [
      // console.log(data),
      // only needed when using shadows - a plane for shadows to drop on
      new PolygonLayer({
        id: 'ground',
        data: landCover,
        stroked: false,
        getPolygon: f => f,
        getFillColor: [0, 0, 0, 0]
      }),
      new GeoJsonLayer({
        id: 'geojson',
        data,
        opacity: 0.9,
        stroked: false,
        filled: true,
        extruded: true,
        wireframe: true,
        getElevation: 8000,
        getFillColor: [65, 182, 196],
        getLineColor: [65, 182, 196],
        pickable: true,
        onHover: this._onHover
      }),
    ];
  }

  _renderTooltip() {
    const { x, y, hoveredObject } = this.state;
    return (
      hoveredObject &&
      (<div className='tooltip' style={{ top: y, left: x }}><div><b>Country</b>
      </div>
        <div>
          <div>{hoveredObject.properties
            .name}</div>
          <div>
            id:{hoveredObject.id}             </div>
        </div>
      </div>));
  }

  render() {
    const { mapStyle = 'https://basemaps.cartocdn.com/gl/dark-matter-nolabels-gl-style/style.json' } = this.props;

    return (
      <DeckGL
        layers={this._renderLayers()}
        effects={this._effects}
        initialViewState={INITIAL_VIEW_STATE}
        controller={true}>
        <StaticMap
          reuseMaps
          mapStyle={mapStyle}
          preventStyleDiffing={true}
          mapboxApiAccessToken={MAPBOX_TOKEN}
        />

        {this._renderTooltip}
      </DeckGL>
    );
  }
}

export function renderToDOM(container) { render(<App />, container); }
