// Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

import {BASEMAP} from '@deck.gl/carto';
import {GeoJsonLayer} from '@deck.gl/layers';
import DeckGL from '@deck.gl/react';
import {tableFromIPC} from 'apache-arrow';
import React from 'react';
import {Button, Card} from 'react-bootstrap';
import {StaticMap} from 'react-map-gl';

import {generateLegend} from '../../components/utils/legend';

export default class CustomChoropleth extends React.Component {
  constructor(props) {
    super(props);
    this.state    = {gjson: [], data: [], clicked: false, current_query: this.props.getquery()};
    this._onHover = this._onHover.bind(this);
    this._onClick = this._onClick.bind(this);
    this._renderTooltip = this._renderTooltip.bind(this);
    this._getFillColor  = this._getFillColor.bind(this);
    this._reset         = this._reset.bind(this);
  }

  _fillColor() {
    if (!this.props.fillcolor) { return this.props.columns[0] }
    return this.props.fillcolor;
  }

  _elevation() {
    if (!this.props.elevation) { return this.props.columns[0] }
    return this.props.elevation;
  }

  componentDidMount() {
    const {geojsonurl = undefined} = this.props;

    if (geojsonurl) {
      fetch(geojsonurl)
        .then(response => response.json())
        .then(({features}) => { this.setState({gjson: features}); })
        .then(() => this._updateLayerData())
        .then((data) => this.setState({data: data}))
        .catch((e) => console.log(e));
    }
    generateLegend(
      '#' + this.props.by + 'legend', this.props.legend_props, this.props.thresholdScale);
  }

  componentDidUpdate() {
    // if parent componet `getquery()` is updated, and is not the same as state.current_query,
    // refetch the data with current query
    if (this.props.getquery() !== this.state.current_query) {
      let updateState = {
        current_query: this.props.getquery()
      }

      if (!this.props.getquery().hasOwnProperty(this.props.by)) {
        // if current chart selections have also been reset
        updateState['clicked'] = false;
      }
      // update state
      this.setState(updateState);
      // update data
      this._updateLayerData()
        .then((data) => this.setState({data: data}))
        .catch((e) => console.log(e));
    }
  }

  _generateApiUrl() {
    const {
      dataset = undefined,
      by      = undefined,
      agg     = undefined,
      columns = undefined  // return all columns
    } = this.props;

    if (!dataset || !by || !agg) { return null; }
    const query = JSON.stringify(this.props.getquery());
    return `http://localhost:3000/api/${dataset}/groupby/${by}/${agg}?columns=${
      columns}&query_dict=${query}`;
  }

  /**
   * transform Arrow.Table to the format
   * {index[0]: {property: propertyValue, ....},
   * index[1]: {property: propertyValue, ....},
   * ...} for easier conversion to geoJSON object
   *
   * @param data Arrow.Table
   * @param by str, column name
   * @param params [{}, {}] result of Arrow.Table.toArray()
   * @returns {index:{props:propsValue}}
   */
  _transformData(data, by, params) {
    return data.reduce((a, v) => {
      a[v[by]] = params.reduce((res, x) => {
        res[x] = v[x];
        return res
      }, {});
      return a;
    }, {});
  }

  /**
   * TODO: pass arrow to binary input to GeoJSONLayer instead of geojson
   * convert an Arrow table to a geoJSON to be consumed by DeckGL GeoJSONLayer. Arrow table results
   * from cudf.DataFrame.toArrow() function.
   *
   * @param table Arrow.Table
   * @param by str, column name to be matched to the geoJSONProperty in the gjson object
   * @param properties [] of column names, properties to add from data to result geoJSON
   * @param geojsonProp str, property name in gjson object to be mapped to `by`
   * @returns geoJSON object consumable by DeckGL GeoJSONLayer
   */
  _convertToGeoJSON(table, by, properties, geojsonProp) {
    const data   = this._transformData(table.toArray(), by, properties);
    let tempjson = [];
    this.state.gjson.forEach((val) => {
      if (val.properties[geojsonProp] in data) {
        tempjson.push({
          type: val.type,
          geometry: val.geometry,
          properties: {...val.properties, ...data[val.properties[geojsonProp]]}
        })
      }
    });
    return tempjson;
  }

  async _updateLayerData() {
    const url =
      this._generateApiUrl();  // `/uber/tracts/groupby/sourceid/mean?columns=travel_time`;
    if (!url) { return null; }
    const table = await tableFromIPC(fetch(url, {method: 'GET'}));
    return this._convertToGeoJSON(table, this.props.by, this.props.columns, this.props.geojsonprop);
  }

  _getFillColor(f) {
    const x  = f?.properties[this._fillColor()];
    const id = f?.properties[this.props.geojsonprop];
    if (x !== undefined && id !== undefined) {
      if (this.state.clicked !== false) {
        if (this.state.clicked !== parseInt(id)) {
          return ([255, 255, 255, 10]);
        } else {
          return this.props.thresholdScale(x);
        }
      } else {
        return this.props.thresholdScale(x);
      }
    }
  }

  _onHover({x, y, object}) { this.setState({x, y, hoveredRegion: object}); }

  _renderTooltip() {
    const {x, y, hoveredRegion} = this.state;
        return (
            hoveredRegion && (
                <div className='deck-tooltip' style={{
      position: 'absolute', left: x, top: y }}>
                    {
                        Object.keys(hoveredRegion.properties).map((value, idx) => {
      return `
                            ${value}: ${hoveredRegion.properties[[value]]}
                            `;
                        })
                    }
                    {/* {hoveredRegion.properties.DISPLAY_NAME} <br />
                Mean Travel Time: {hoveredRegion.properties.travel_time} */}
                </div >)
        );
    }

    _onClick(f) {
        const id = f?.object?.properties[this.props.geojsonprop];
        if (this.state.clicked !== parseInt(id)) {
            this.props.updatequery({ [this.props.by]: parseInt(id) });
            this.setState({ clicked: parseInt(id) });
        } else {
            //double click deselects
            this._reset();
  }
}

_renderLayers() {
  const {data} = this.state;

  return [new GeoJsonLayer({
    data,
    highlightColor: [200, 200, 200, 200],
    autoHighlight: true,
    wireframe: false,
    pickable: true,
    stroked: false,  // only on extrude false
    filled: true,
    extruded: true,
    lineWidthScale: 10,  // only if extrude false
    lineWidthMinPixels: 1,
    getRadius: 100,  // only if points
    getLineWidth: 10,
    opacity: 50,
    getFillColor: this._getFillColor,
    getElevation: f => f?.properties[this._elevation()] * 5,
    onClick: this._onClick,
    onHover: this._onHover,
    getLineColor: [0, 188, 212, 100],
    updateTriggers:
      {getFillColor: [this.state.clicked, this.state.data], getElevation: [this.state.data]}
  })]
}

_reset() {
  this.props.updatequery({[this.props.by]: undefined});
  this.setState({clicked: false})
}

render() {
        return (
            <Card border='light' bg='secondary' text='light' className='text-center'>
                <Card.Header className='h5'>Trip {this.props.by} aggregated by {this.props.agg}
                    <Button variant='primary' size='sm' onClick={this._reset} className='float-sm-right'> Reset </Button>
                </Card.Header>
                <Card.Body>
                    <svg className='legend float-left' id={this.props.by + 'legend'} height='110' width='200' style={{
    'zIndex': 1, 'position': 'relative' }}></svg>
                    <DeckGL
                        layers={this._renderLayers()}
                        initialViewState={this.props.initialviewstate}
                        controller={true} style={{ "zIndex": 0 }}
                    >
                        <StaticMap mapStyle={BASEMAP.DARK_MATTER} />
                        {this._renderTooltip}
                    </DeckGL>
                </Card.Body>
            </Card >
        )
    }
}
