// Copyright (c) 2022, NVIDIA CORPORATION.
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

import {log as deckLog} from '@deck.gl/core';
import {ArcLayer, GeoJsonLayer} from '@deck.gl/layers';
import {MapboxOverlay} from '@deck.gl/mapbox';
import {scaleQuantile} from 'd3-scale';
import * as maplibre from 'maplibre-gl';

deckLog.level = 0;
deckLog.enable(false);

const mapStyle = 'https://basemaps.cartocdn.com/gl/dark-matter-nolabels-gl-style/style.json';
const DATA_URL =
  'https://raw.githubusercontent.com/uber-common/deck.gl-data/master/examples/arc/counties.json';

const INITIAL_VIEW_STATE = {
  longitude: -100,
  latitude: 40.7,
  zoom: 3,
  maxZoom: 15,
  pitch: 30,
  bearing: 30,
};

// Add MapLibre GL for the basemap
const map = new maplibre.Map({
  style: mapStyle,
  interactive: true,
  container: document.body,
  zoom: INITIAL_VIEW_STATE.zoom,
  pitch: INITIAL_VIEW_STATE.pitch,
  bearing: INITIAL_VIEW_STATE.bearing,
  maxZoom: INITIAL_VIEW_STATE.maxZoom,
  center: [
    INITIAL_VIEW_STATE.longitude,
    INITIAL_VIEW_STATE.latitude,
  ],
});

map.scrollZoom.setWheelZoomRate(1 / 25);

const deck = new MapboxOverlay({layers: [], interleaved: true});

map.addControl(deck);

fetch(DATA_URL)
  .then(response => response.json())
  .then(({features}) => getLayers(calcArcs(features)))  //
  .then((props) => deck.setProps(props));

function calcArcs(data, selectedCounty = deck._props.selectedCounty) {
  if (!data || !data.length) { return; }
  if (!selectedCounty) {  //
    selectedCounty = data.find(f => f.properties.name === 'Los Angeles, CA');
  }
  const {flows, centroid} = selectedCounty.properties;

  const arcs = Object.keys(flows).map(toId => {
    const f = data[toId];
    return {
      source: centroid,
      target: f.properties.centroid,
      value: flows[toId],
    };
  });

  const scale =
    scaleQuantile().domain(arcs.map(a => Math.abs(a.value))).range(inFlowColors.map((c, i) => i));

  arcs.forEach(a => {
    a.gain     = Math.sign(a.value);
    a.quantile = scale(Math.abs(a.value));
  });

  return {data, arcs, selectedCounty};
}

function getLayers({data, arcs, ...props}) {
  return {
    ...props,
    layers: [
      new GeoJsonLayer({
        id: 'geojson',
        data: data,
        filled: true,
        stroked: true,
        pickable: true,
        autoHighlight: true,
        lineWidthMinPixels: 1,
        getLineWidth: 1,
        getFillColor: [0, 0, 0, 0],
        getLineColor: [255, 255, 255, 135],
        onHover: ({x, y, object}) => deck.setProps({x, y, hoveredCounty: object}),
        onClick: ({object})       => deck.setProps(getLayers(calcArcs(data, object))),
      }),
      new ArcLayer({
        id: 'arc',
        data: arcs,
        getWidth: 2,
        getSourcePosition: d => d.source,
        getTargetPosition: d => d.target,
        getSourceColor: d    => (d.gain > 0 ? inFlowColors : outFlowColors)[d.quantile],
        getTargetColor: d    => (d.gain > 0 ? outFlowColors : inFlowColors)[d.quantile],
      }),
    ]
  };
}

const inFlowColors = [
  [255, 255, 204],
  [199, 233, 180],
  [127, 205, 187],
  [65, 182, 196],
  [29, 145, 192],
  [34, 94, 168],
  [12, 44, 132]
];

const outFlowColors = [
  [255, 255, 178],
  [254, 217, 118],
  [254, 178, 76],
  [253, 141, 60],
  [252, 78, 42],
  [227, 26, 28],
  [177, 0, 38]
];
