import Head from 'next/head';
import React from 'react';
import CustomChoropleth from '../../components/charts/deck.geo';
import CustomBar from '../../components/charts/echarts.bar';
import Indicator from '../../components/charts/indicator';
import { Row, Col, Container } from 'react-bootstrap';
import Layout from '../../components/layout';

export default class FirstPost extends React.Component {
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
                    <title>First Post</title>
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
                                        columns="travel_time"
                                        geojsonurl="http://localhost:3000/data/san_francisco_censustracts.geojson"
                                        geojsonprop="MOVEMENT_ID"
                                        initialviewstate={{ longitude: -122, latitude: 37, zoom: 6, maxZoom: 16, pitch: 0, bearing: 0 }}
                                        getquery={this._getQuery}
                                        updatequery={this._updateQuery}
                                    ></CustomChoropleth>
                                </Col>
                                <Col md={6}>
                                    <CustomChoropleth
                                        dataset="uber"
                                        by="dstid"
                                        agg="mean"
                                        columns="travel_time"
                                        geojsonurl="http://localhost:3000/data/san_francisco_censustracts.geojson"
                                        geojsonprop="MOVEMENT_ID"
                                        initialviewstate={{ longitude: -122, latitude: 37, zoom: 6, maxZoom: 16, pitch: 0, bearing: 0 }}
                                        getquery={this._getQuery}
                                        updatequery={this._updateQuery}
                                    ></CustomChoropleth>
                                </Col>
                            </Row>
                        </Col>
                    </Row>
                </Container>
            </Layout>
        )
    }
}
