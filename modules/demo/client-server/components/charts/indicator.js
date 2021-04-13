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

import React from 'react';
import * as d3 from "d3";
import { Card } from 'react-bootstrap';

export default class Indicator extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      current_query: this.props.getquery(),
      size: 0
    }
  }

  componentDidMount() {
    this._updateSize()
      .then((data) => this.setState({ size: d3.format(',')(data) }))
      .catch((e) => console.log(e));
  }

  componentDidUpdate() {
    // if parent componet `getquery()` is updated, and is not the same as state.current_query, 
    // refetch the data with current query
    if (this.props.getquery() !== this.state.current_query) {
      // update state
      this.setState({ current_query: this.props.getquery() });
      // update data
      this._updateSize()
        .then((data) => this.setState({ size: d3.format(',')(data) }))
        .catch((e) => console.log(e));
    }
  }

  _generateApiUrl() {
    const {
      dataset = undefined,
    } = this.props;

    if (!dataset) { return null; }
    const query = JSON.stringify(this.props.getquery());
    return `http://localhost:3000/api/${dataset}/numRows?query_dict=${query}`;
  }

  async _updateSize() {
    const url = this._generateApiUrl(); // `/uber/tracts/groupby/sourceid/mean?columns=travel_time`;
    if (!url) { return null; }
    const size = await fetch(url, { method: 'GET' }).then((res) => res.json()).then(data => { return data });
    return size;
  }

  render() {
    return (
      <Card border="light" bg="secondary" text="light" className="text-center mb-3">
        <Card.Header className="h5"> Data Points Selected
        </Card.Header>
        <Card.Body>
          <span className="h4 text-center">{this.state.size}</span>
        </Card.Body>
      </Card>
    )
  }
}
