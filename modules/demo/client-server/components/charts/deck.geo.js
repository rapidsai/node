import {GeoJsonLayer} from '@deck.gl/layers';
import DeckGL from '@deck.gl/react';
import React from 'react';
import { StaticMap } from 'react-map-gl';
import {BASEMAP} from '@deck.gl/carto';
import { Table } from 'apache-arrow';
import * as d3 from "d3";
import { Button, Card} from 'react-bootstrap';

// color scale for choropleth charts
const COLOR_SCALE = [
    [49, 130, 189, 100],
    [107, 174, 214, 100],
    [123, 142, 216, 100],
    [226, 103, 152, 100],
    [255, 0, 104, 100],
  ];
const thresholdScale =
    d3.scaleThreshold().domain([0, 400, 800, 1000, 2000, 4000]).range(COLOR_SCALE);

export default class CustomChoropleth extends React.Component {
    constructor(props) {
        super(props);
        const { geojsonurl = undefined } = this.props;
        
        this.state = {
            gjson: [],
            data: [],
            clicked: false
        }
        this._onHover = this._onHover.bind(this);
        this._onClick = this._onClick.bind(this);
        this._renderTooltip = this._renderTooltip.bind(this);
        this._getFillColor = this._getFillColor.bind(this);
        this._reset = this._reset.bind(this);
        
        if(geojsonurl){
            fetch(geojsonurl)
                .then(response => response.json())
                .then(({features}) => {
                    this.setState({gjson: features});
                })
                .then(() => this._updateLayerData())
                .then((data) => this.setState({data: data}))
                .catch((e) => console.log(e));
        }
    }
    _generateApiUrl(){
        const {
            dataset = undefined,
            by = undefined,
            agg = undefined,
            columns = undefined //return all columns
        } = this.props;

        if(!dataset || !by || !agg){ return null;}
        return `http://localhost:5000/${dataset}/tracts/groupby/${by}/${agg}?columns=${columns}`;
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
            a[v[[by]]] = params.reduce((res, x) => {
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
        const data = this._transformData(table.toArray(), by, properties);
        let tempjson = [];
        this.state.gjson.forEach((val) => {
        if (val.properties[geojsonProp] in data) {
            tempjson.push({
            type: val.type,
            geometry: val.geometry,
            properties: { ...val.properties, ...data[val.properties[geojsonProp]] }
            })
        }
        });
        return tempjson;
    }  

    async _updateLayerData(){
        const url = this._generateApiUrl(); // `/uber/tracts/groupby/sourceid/mean?columns=travel_time`;
        if (!url) { return null; }
        const table = await Table.from(fetch(url, {method: 'GET'}));
        return this._convertToGeoJSON(table, this.props.by, [this.props.columns], this.props.geojsonprop);
    }


    _getFillColor(f){
        const x = f?.properties?.travel_time;
        const id = f?.properties?.MOVEMENT_ID;
        if (x !== undefined && id !== undefined) {
            if (this.state.clicked !== false) {
              if (this.state.clicked !== parseInt(id)) {
                return ([255, 255, 255, 10]);
              } else {
                return thresholdScale(x);
              }
            } else {
              return thresholdScale(x);
            }
          }
    }

    _onHover({x, y, object}){
        this.setState({x, y, hoveredCensusTract: object});
    }

    _renderTooltip(){
        const { x, y, hoveredCensusTract } = this.state;
        
        return (
            hoveredCensusTract && (
                <div className="deck-tooltip" style={{position: 'absolute', left: x, top: y}}>
                {hoveredCensusTract.properties.DISPLAY_NAME} <br/>
                Mean Travel Time: {hoveredCensusTract.properties.travel_time}
            </div>)
        );
    }

    _onClick(f){
        const id = f?.object?.properties?.MOVEMENT_ID;
        console.log(id);
        if(this.state.clicked !== parseInt(id)){
            this.setState({clicked: parseInt(id)});
        }else{
            //double click deselects
            this.setState({clicked: false});
        }
    }

    _renderLayers(){
        const { data } = this.state;

        return [
            new GeoJsonLayer({
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
                getFillColor:this._getFillColor,
                getElevation: f => f?.properties?.travel_time * 10,
                onClick: this._onClick,
                onHover: this._onHover,
                getLineColor: [0, 188, 212, 100],
                updateTriggers: {
                    getFillColor: [this.state.clicked]
                }
            })
        ]
    }

    _reset(){
        this.setState({
            clicked: false
        })
    }

    render() {
        return (
            <Card border="dark">
                <Card.Header>{this.props.by} choropleth
                <Button variant="primary" onClick={this._reset}> Reset </Button>
                </Card.Header>
                <Card.Body>
                    <DeckGL
                        layers={this._renderLayers()}
                        initialViewState={this.props.initialviewstate}
                        controller={true}
                    >
                       <StaticMap mapStyle={BASEMAP.DARK_MATTER} />
                     {this._renderTooltip}
                    </DeckGL>
                </Card.Body>
            </Card>
        )
    }
}
