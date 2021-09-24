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
import { Table } from 'apache-arrow';
import Paper from '@material-ui/core/Paper';
import MatTable from '@material-ui/core/Table';
import TableBody from '@material-ui/core/TableBody';
import TableCell from '@material-ui/core/TableCell';
import TableContainer from '@material-ui/core/TableContainer';
import TableHead from '@material-ui/core/TableHead';
import TablePagination from '@material-ui/core/TablePagination';
import TableRow from '@material-ui/core/TableRow';
import Typography from '@material-ui/core/Typography';

const columns = [
  { id: 'id', label: 'ID', minWidth: 0, },
  { id: 'revid', label: 'Rev ID', minWidth: 0, },
  { id: 'url', label: 'URL', minWidth: 0, },
  { id: 'title', label: 'Title', minWidth: 0, },
  { id: 'text', label: 'Text', minWidth: 500 }
];

function createData(id, revid, url, title, text) {
  return { id, revid, url, title, text };
}

function formatData(data) {
  if (Object.keys(data).length === 0) {
    return [];
  }

  let rows = [];
  const length = data['title'].length;
  for (let i = 0; i < length; ++i) {
    const id = data['id'][i];
    const revid = data['revid'][i];
    const url = data['url'][i];
    const title = data['title'][i];
    const text = data['text'][i];
    rows.push(
      createData(
        id,
        revid,
        url,
        title,
        text
      )
    );

    // Let's just process the first 500 results due to client side rendering bottleneck.
    if (i >= 500) {
      break;
    }
  }

  return rows;
}

export class QueryDashboard extends React.Component {
  constructor() {
    super();
    this.state = {
      query: '',
      queryTime: '',
      queryResult: [],
      queryButtonEnabled: false,
      page: 0,
      rowsPerPage: 10,
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
          queryResult: formatData(result),
          queryTime: table.schema.metadata.get('queryTime'),
          resultCount: table.length,
          page: 0,
          rowsPerPage: 10
        });
      });
      this.setState({ queryButtonEnabled: true });
    }
  }

  handleChangePage = (event, newPage) => {
    this.setState({
      page: newPage
    });
  };

  handleChangeRowsPerPage = (event) => {
    this.setState({
      rowsPerPage: +event.target.value,
      page: 0
    });
  };

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
        <Paper>
          <Typography style={{ marginLeft: 5 }} variant="h6" id="tableTitle" component="div">
            <div>Query Time: {Math.round(this.state.queryTime)} ms</div>
            <div>Results: {this.state.resultCount ?? 0}</div>
          </Typography>
          <TableContainer>
            <MatTable stickyHeader aria-label="sticky table">
              <TableHead>
                <TableRow>
                  {columns.map((column) => (
                    <TableCell
                      key={column.id}
                      align={column.align}
                      style={{ minWidth: column.minWidth }}
                    >
                      {column.label}
                    </TableCell>
                  ))}
                </TableRow>
              </TableHead>
              <TableBody>
                {this.state.queryResult.slice(this.state.page * this.state.rowsPerPage, this.state.page * this.state.rowsPerPage + this.state.rowsPerPage).map((row) => {
                  return (
                    <TableRow hover role="checkbox" tabIndex={-1} key={row.code}>
                      {columns.map((column) => {
                        const value = row[column.id];
                        return (
                          <TableCell key={column.id} align={column.align}>
                            <div style={{ display: 'block', textOverflow: 'ellipsis', overflow: 'hidden', maxHeight: 100 }}>
                              {column.format && typeof value === 'number' ? column.format(value) : value}
                            </div>
                          </TableCell>
                        );
                      })}
                    </TableRow>
                  );
                })}
              </TableBody>
            </MatTable>
          </TableContainer>
          <TablePagination
            rowsPerPageOptions={[10, 25, 100, 500]}
            component="div"
            count={this.state.queryResult.length}
            rowsPerPage={this.state.rowsPerPage}
            page={this.state.page}
            onPageChange={this.handleChangePage}
            onRowsPerPageChange={this.handleChangeRowsPerPage}
          />
        </Paper>
      </Container >
    )
  }
}
