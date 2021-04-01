import Head from 'next/head';
import React from 'react';
import CustomChoropleth from '../../components/charts/deck.geo';
import { Row, Col, Container } from 'react-bootstrap';
import Layout from '../../components/layout';

export default class FirstPost extends React.Component {
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
                            ></CustomChoropleth>
                        </Col>
                    </Row>
                </Container>
            </Layout>
        )
    }
}
