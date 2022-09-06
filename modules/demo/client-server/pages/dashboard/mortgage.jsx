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

import * as d3 from 'd3';
import Head from 'next/head';
import * as React from 'react';
import { Col, Container, Row } from 'react-bootstrap';

import CustomChoropleth from '../../components/charts/deck.geo.jsx';
import CustomBar from '../../components/charts/echarts.bar.jsx';
import Indicator from '../../components/charts/indicator.jsx';
import Layout from '../../components/layout';

// color scale for choropleth charts
const COLOR_SCALE = [
  [49, 130, 189, 100],
  [107, 174, 214, 100],
  [123, 142, 216, 100],
  [226, 103, 152, 100],
  [255, 0, 104, 100],
];
const thresholdScale =
  d3.scaleThreshold().domain([0, 0.196, 0.198, 0.200, 0.202]).range(COLOR_SCALE);

export default class MortgageDashboard extends React.Component {
  constructor(props) {
    super(props);
    this.state = { query_dict: {} };
    this._getQuery = this._getQuery.bind(this);
    this._resetQuery = this._resetQuery.bind(this);
    this._updateQuery = this._updateQuery.bind(this);
  }

  _getQuery() { return this.state.query_dict; }

  _resetQuery() { this.setState({ query_dict: {} }); }

  _updateQuery(query_dict) {
    query_dict = { ...this.state.query_dict, ...query_dict };
    this.setState({ query_dict: query_dict })
  }

  render() {
    return (
      <Layout title='Fannie Mae Mortgage Dashboard' resetall={this._resetQuery}>
        <Head>
          <title>Fannie Mae Mortgage Dashboard</title>
        </Head>
        <Container fluid>
          <Row>
            <Col md={3}>
              <Col>
                <Indicator
                  dataset='mortgage'
                  getquery={this._getQuery}
                  updatequery={this._updateQuery} />
              </Col>
              <Col>
                <CustomBar
                  dataset='mortgage'
                  x='dti'
                  y='zip'
                  agg='count'
                  getquery={this._getQuery}
                  updatequery={this._updateQuery}
                  className='mb-3'
                />
              </Col>
              <Col>
                <CustomBar
                  dataset='mortgage'
                  x='borrower_credit_score'
                  y='zip'
                  agg='count'
                  getquery={this._getQuery}
                  updatequery={this._updateQuery}
                />
              </Col>
            </Col>
            <Col md={9}>
              <CustomChoropleth
                dataset="mortgage"
                by="zip"
                agg="mean"
                columns={['delinquency_12_prediction', 'current_actual_upb', 'dti', 'borrower_credit_score']}
                fillcolor="delinquency_12_prediction"
                elevation="current_actual_upb"
                geojsonurl="https://raw.githubusercontent.com/rapidsai/cuxfilter/GTC-2018-mortgage-visualization/javascript/demos/GTC%20demo/src/data/zip3-ms-rhs-lessprops.json"
                geojsonprop='ZIP3'
                initialviewstate={{
                  longitude: -101,
                  latitude: 37,
                  zoom: 3,
                  maxZoom: 16,
                  pitch: 0,
                  bearing: 0
                }}
                getquery={this._getQuery}
                updatequery={this._updateQuery}
                thresholdScale={thresholdScale}
                legend_props={{
                  '0.100': '0 to 0.196',
                  '0.197': '0.196 to 0.198',
                  '0.199': '0.198 to 0.200',
                  '0.201': '0.200 to 0.202',
                  '0.203': '0.202 to 0.2+'
                }}
              />
            </Col>
          </Row>
        </Container>
      </Layout>
    )
  }
}
