import Head from 'next/head';
import React from 'react';
import CustomChoropleth from '../../components/charts/deck.geo';
import { Row, Col, Container } from 'react-bootstrap';
import Layout from '../../components/layout';

export default class FirstPost extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            query_dict: {}
        }
        this._getQuery = this._getQuery.bind(this);
        this._setQuery = this._setQuery.bind(this);
        this._updateQuery = this._updateQuery.bind(this);
    }

    _getQuery() {
        return this.state.query_dict;
    }

    _setQuery(query_dict) {
        this.setState({
            query_dict: query_dict
        });
    }

    _updateQuery(query_dict) {
        query_dict = { ...this.state.query_dict, ...query_dict }
        console.log("updated");
        this.setState({
            query_dict: query_dict
        })
    }

    render() {
        return (
            <Layout title="Uber Dashboard">
                <Head>
                    <title>First Post</title>
                </Head>
                <Container fluid>
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
                </Container>
            </Layout>
        )
    }
}
