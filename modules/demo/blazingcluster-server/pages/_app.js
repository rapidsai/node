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

import 'bootstrap/dist/css/bootstrap.min.css';
import './style.css';

import { Container, Navbar, Nav, Row, Col, FormControl } from 'react-bootstrap';
import { QueryBuilder } from '../components/querybuilder';
import Button from '@material-ui/core/Button';
import React from 'react';

export default function App() {
  return (
    <div>
      <Navbar bg="dark" variant="dark">
        <Container>
          <Navbar.Brand className={"navbar"}>node-rapids ┆ Blazing Cluster Server Demo</Navbar.Brand>
          <Nav>
            <Nav.Link href="https://github.com/rapidsai/node">node-rapids github</Nav.Link>
          </Nav>
        </Container>
      </Navbar>
      <QueryDashboard />
    </div >
  )
}

class QueryDashboard extends React.Component {
  constructor() {
    super();
    this.state = {
      query: ''
    };

    this.runQuery = this.runQuery.bind(this);
  }

  onQueryChange = (updatedQuery) => {
    this.setState({ query: this._parseQuery(updatedQuery) });
  }

  async runQuery() {
    if (this.state.query.length) {
      await fetch('http://localhost:3000/run_query').then(response => response.json()).then(data => { console.log(data) });
    }
  }

  _parseQuery(query) {
    if (query === undefined || query.length == 0) return '';
    return `SELECT ${JSON.parse(query)} FROM test_table`;
  }

  render() {
    return (
      <Container style={{ paddingTop: 10 }}>
        <Row className={"justify-content-center"}>
          <Col lg={8} md={8} sm={8} className={"customCol"}>
            <QueryBuilder onQueryChange={this.onQueryChange} />
          </Col>
          <Col className={"customCol"} lg md sm xs={12}>
            <Button variant="contained" color="primary" className={"queryButton"} disabled={!this.state.query.length} onClick={this.runQuery}>Run Query</Button>
            <FormControl style={{ marginTop: 20 }} rows="4" as="textarea" disabled={true} value={"Waiting for query result..."} aria-label="SQL result" />
          </Col>
        </Row>
      </Container>
    )
  }
}
