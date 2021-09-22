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

import Button from '@material-ui/core/Button';
import React from 'react';
import { Col, Container, FormControl, InputGroup, Row } from 'react-bootstrap';
import { QueryBuilder } from './querybuilder';
import { QueryResultTable } from './queryresult';
import { Table } from 'apache-arrow';

export class QueryDashboard extends React.Component {
  constructor() {
    super();
    this.state = {
      query: '',
      queryTime: '',
      queryResult: {},
      queryButtonEnabled: false,
    };

    this.runQuery = this.runQuery.bind(this);
  }

  onQueryChange = (updatedQuery) => {
    this.setState({ query: updatedQuery, queryButtonEnabled: updatedQuery.length });
  }

  async runQuery() {
    if (this.state.queryButtonEnabled) {
      this.setState({ queryButtonEnabled: false });
      await fetch(`/run_query`, {
        method: `POST`,
        headers: {
          'accepts': `application/octet-stream`
        },
        body: `${this.state.query}`
      }).then((res) => Table.from(res)).then((table) => {
        const result = table.length == 0 ? {} : {
          id: [...table.getColumn("id")].map((x) => +x),
          revid: [...table.getColumn("revid")].map((x) => +x),
          url: [...table.getColumn("url")],
          title: [...table.getColumn("title")],
          text: [...table.getColumn("text")],
        };
        this.setState({
          queryResult: result,
          queryTime: table.schema.metadata.get('queryTime'),
          resultCount: table.length,
        });
      });
      this.setState({ queryButtonEnabled: true });
    }
  }

  render() {
    return (
      <Container style={{
        paddingTop: 40
      }}>
        <QueryBuilder onQueryChange={
          this.onQueryChange} />
        <Row style={{ marginTop: 20, marginBottom: 20 }}>
          <Col lg={9} md={9} sm={8} xs={8}>
            <InputGroup className={"queryInput"}>
              <div className="input-group-prepend">
                <span className="input-group-text">Query: </span>
              </div>
              <FormControl className={"queryInput"} value={this.state.query} disabled={true} type={"text"} />
            </InputGroup>
          </Col>
          <Col>
            <Button variant='contained' color='primary' className={'queryButton'} disabled={!this.state.queryButtonEnabled} onClick={this.runQuery}>Run Query</Button>
          </Col>
        </Row>
        <QueryResultTable data={this.state.queryResult} queryTime={this.state.queryTime} resultCount={this.state.resultCount} />
      </Container >
    )
  }
}
