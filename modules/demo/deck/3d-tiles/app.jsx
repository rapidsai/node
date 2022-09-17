import { Tile3DLayer } from '@deck.gl/geo-layers';
import DeckGL from '@deck.gl/react';
import { CesiumIonLoader } from '@loaders.gl/3d-tiles';
import { registerLoaders } from '@loaders.gl/core';
// To manage dependencies and bundle size, the app must decide which supporting loaders to bring in
import { DracoWorkerLoader } from '@loaders.gl/draco';
import * as React from 'react';
import { Component } from 'react';
import { render } from 'react-dom';
import { StaticMap } from 'react-map-gl';

registerLoaders([DracoWorkerLoader]);

// Set your mapbox token here
const MAPBOX_TOKEN = process.env.MapboxAccessToken;  // eslint-disable-line

const ION_ASSET_ID = 43978;
const ION_TOKEN =
  'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiJlYWMxMzcyYy0zZjJkLTQwODctODNlNi01MDRkZmMzMjIxOWIiLCJpZCI6OTYyMCwic2NvcGVzIjpbImFzbCIsImFzciIsImdjIl0sImlhdCI6MTU2Mjg2NjI3M30.1FNiClUyk00YH_nWfSGpiQAjR5V2OvREDq1PJ5QMjWQ';
const TILESET_URL = `https://assets.cesium.com/${ION_ASSET_ID}/tileset.json`;

const INITIAL_VIEW_STATE = {
  latitude: 40,
  longitude: -75,
  pitch: 45,
  maxPitch: 60,
  bearing: 0,
  minZoom: 2,
  maxZoom: 30,
  zoom: 17
};

export default class App extends Component {
  constructor(props) {
    super(props);
    this.state = { initialViewState: INITIAL_VIEW_STATE, attributions: [] };
  }

  _onTilesetLoad(tileset) {
    this._centerViewOnTileset(tileset);
    if (this.props.updateAttributions) {
      this.props.updateAttributions(tileset.credits && tileset.credits.attributions);
    }
  }

  // Recenter view to cover the new tileset, with a fly-to transition
  _centerViewOnTileset(tileset) {
    const { cartographicCenter, zoom } = tileset;
    this.setState({
      initialViewState: {
        ...INITIAL_VIEW_STATE,

        // Update deck.gl viewState, moving the camera to the new tileset
        longitude: cartographicCenter[0],
        latitude: cartographicCenter[1],
        zoom,
        bearing: INITIAL_VIEW_STATE.bearing,
        pitch: INITIAL_VIEW_STATE.pitch
      }
    });
  }

  _renderTile3DLayer() {
    return new Tile3DLayer({
      id: 'tile-3d-layer',
      pointSize: 2,
      data: TILESET_URL,
      loader: CesiumIonLoader,
      loadOptions: { 'cesium-ion': { accessToken: ION_TOKEN } },
      onTilesetLoad: this._onTilesetLoad.bind(this)
    });
  }

  render() {
    const { initialViewState } = this.state;
    const tile3DLayer = this._renderTile3DLayer();
    const { mapStyle = 'mapbox://styles/uberdata/cive485h000192imn6c6cc8fc' } = this.props;

    return (
      <div>
        <DeckGL layers={[tile3DLayer]} initialViewState={initialViewState} controller={true}>
          <StaticMap mapStyle={mapStyle} mapboxApiAccessToken={MAPBOX_TOKEN} preventStyleDiffing />
        </DeckGL>
      </div>);
  }
}

export function renderToDOM(container) { render(<App />, container); }
