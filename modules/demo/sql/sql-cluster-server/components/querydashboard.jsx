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
import LoadingOverlay from 'react-loading-overlay';

const MAX_RESULTS_TO_DISPLAY = 500;
const columns = [
  { id: 'id', label: 'ID', minWidth: 0, },
  { id: 'revid', label: 'Rev ID', minWidth: 0, },
  { id: 'url', label: 'URL', minWidth: 0, },
  { id: 'title', label: 'Title', minWidth: 0, },
  { id: 'text', label: 'Text', minWidth: 500 }
];

function formatData(table) {
  let rows = [];
  if (table.length == 0) {
    return rows;
  }

  const resultsToDisplay = table.length < MAX_RESULTS_TO_DISPLAY ? table.length : MAX_RESULTS_TO_DISPLAY;
  const ids = [...table.getColumn("id")].map((x) => +x).slice(0, resultsToDisplay);
  const revids = [...table.getColumn("revid")].map((x) => +x).slice(0, resultsToDisplay);
  const urls = [...table.getColumn("url")].slice(0, resultsToDisplay);
  const titles = [...table.getColumn("title")].slice(0, resultsToDisplay);
  const texts = [...table.getColumn("text")].slice(0, resultsToDisplay);

  for (let i = 0; i < resultsToDisplay; ++i) {
    rows.push({
      id: ids[i],
      revid: revids[i],
      url: urls[i],
      title: titles[i],
      text: texts[i]
    });
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
      runningQuery: false,
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
      this.setState({ runningQuery: true });
      await fetch(`/run_query`, {
        method: `POST`,
        headers: {
          'accepts': `application/octet-stream`
        },
        body: `${this.state.query}`
      }).then((res) => Table.from(res)).then((table) => {
        this.setState({
          queryResult: formatData(table),
          queryTime: table.schema.metadata.get('queryTime'),
          resultCount: table.schema.metadata.get('queryResults'),
          page: 0,
          rowsPerPage: 10
        });
      });
      this.setState({ runningQuery: false });
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
        <LoadingOverlay
          active={this.state.runningQuery}
          spinner
          text='Running query...'
        >
          <QueryBuilder onQueryChange={
            this.onQueryChange} />
        </LoadingOverlay>
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
          <div style={{ textAlign: "end", paddingRight: "20px", paddingBottom: "5px", fontSize: "12px", color: "grey" }}>
            (Table only displays 500 results max)
          </div>
        </Paper>
      </Container >
    )
  }
}
