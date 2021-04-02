// Copyright (c) 2021, NVIDIA CORPORATION.
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

import Head from 'next/head';
import React from 'react';
import CustomChoropleth from '../../components/charts/deck.geo';
import CustomBar from '../../components/charts/echarts.bar';
import Indicator from '../../components/charts/indicator';
import { Row, Col, Container } from 'react-bootstrap';
import Layout from '../../components/layout';
import * as d3 from "d3";

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

export default class UberDashboard extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            query_dict: {}
        }
        this._getQuery = this._getQuery.bind(this);
        this._resetQuery = this._resetQuery.bind(this);
        this._updateQuery = this._updateQuery.bind(this);
    }

    _getQuery() {
        return this.state.query_dict;
    }

    _resetQuery() {
        this.setState({
            query_dict: {}
        });
    }

    _updateQuery(query_dict) {
        query_dict = { ...this.state.query_dict, ...query_dict }
        this.setState({
            query_dict: query_dict
        })
    }

    render() {
        return (
            <Layout title="Uber Dashboard" resetall={this._resetQuery}>
                <Head>
                    <title>Uber Dashboard</title>
                </Head>
                <Container fluid>
                    <Row>
                        <Col md={3}>
                            <Col>
                                <Indicator
                                    dataset="uber"
                                    getquery={this._getQuery}
                                    updatequery={this._updateQuery}
                                />
                            </Col>
                            <Col>
                                <CustomBar
                                    dataset="uber"
                                    x="day"
                                    y="sourceid"
                                    agg="count"
                                    getquery={this._getQuery}
                                    updatequery={this._updateQuery}
                                    xaxisdata={Array.from({ length: 31 }, (v, k) => k + 1)}
                                    className="mb-3"
                                ></CustomBar>
                            </Col>
                            <Col>
                                <CustomBar
                                    dataset="uber"
                                    x="start_hour"
                                    y="sourceid"
                                    agg="count"
                                    getquery={this._getQuery}
                                    updatequery={this._updateQuery}
                                    xaxisdata={['AM Peak', 'Midday', 'PM Peak', 'Evening', 'Early Morning']}
                                ></CustomBar>
                            </Col>
                        </Col>
                        <Col md={9}>
                            <Row>
                                <Col md={6}>
                                    <CustomChoropleth
                                        dataset="uber"
                                        by="sourceid"
                                        agg="mean"
                                        columns={["travel_time"]}
                                        geojsonurl="http://localhost:3000/data/san_francisco_censustracts.geojson"
                                        geojsonprop="MOVEMENT_ID"
                                        initialviewstate={{ longitude: -122, latitude: 37, zoom: 6, maxZoom: 16, pitch: 0, bearing: 0 }}
                                        getquery={this._getQuery}
                                        updatequery={this._updateQuery}
                                        thresholdScale={thresholdScale}
                                        legend_props={{
                                            0: '0s to 400s',
                                            500: '400s to 800s',
                                            900: '800s to 1000s',
                                            1100: '1000s or higher',
                                        }}

                                    ></CustomChoropleth>
                                </Col>
                                <Col md={6}>
                                    <CustomChoropleth
                                        dataset="uber"
                                        by="dstid"
                                        agg="mean"
                                        columns={["travel_time", "DISPLAY_NAME"]}
                                        geojsonurl="http://localhost:3000/data/san_francisco_censustracts.geojson"
                                        geojsonprop="MOVEMENT_ID"
                                        initialviewstate={{ longitude: -122, latitude: 37, zoom: 6, maxZoom: 16, pitch: 0, bearing: 0 }}
                                        getquery={this._getQuery}
                                        updatequery={this._updateQuery}
                                        thresholdScale={thresholdScale}
                                        legend_props={{
                                            0: '0s to 400s',
                                            500: '400s to 800s',
                                            900: '800s to 1000s',
                                            1100: '1000s or higher',
                                        }}

                                    ></CustomChoropleth>
                                </Col>
                            </Row>
                        </Col>
                    </Row>
                </Container >
            </Layout >
        )
    }
}
