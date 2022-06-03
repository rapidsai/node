// Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

import { tableFromIPC } from 'apache-arrow';
import * as d3 from 'd3';
import ReactECharts from 'echarts-for-react';
import React from 'react';
import { Card } from 'react-bootstrap';

export default class CustomBar extends React.Component {
  constructor(props) {
    super(props);
    this.state = { options: {}, current_query: this.props.getquery(), selected_range: false };
    this.echartRef = undefined;
    this._getOptions = this._getOptions.bind(this);
    this._onBrushSelected = this._onBrushSelected.bind(this);
    this._onBrushReset = this._onBrushReset.bind(this);
  }

  componentDidMount() {
    this._updateData()
      .then((data) => this.setState({ options: this._getOptions(data) }))
      .catch((e) => console.log(e));
  }

  componentDidUpdate() {
    // if parent componet `getquery()` is updated, and is not the same as state.current_query,
    // refetch the data with current query
    if (this.props.getquery() !== this.state.current_query) {
      let updateState = {
        current_query: this.props.getquery()
      }

      if (!this.props.getquery().hasOwnProperty(this.props.x)) {
        // if current chart selections have also been reset
        updateState['selected_range'] = false;
        // clear brush
        if (this.echartRef) {
          this.echartRef.getEchartsInstance().dispatchAction({ type: 'brush', areas: [] });
        }
      }
      // update state
      this.setState(updateState);
      // update data
      this._updateData()
        .then((data) => this.setState({ options: this._getOptions(data) }))
        .catch((e) => console.log(e));
    }
  }

  _generateApiUrl() {
    const {
      dataset = undefined,
      x = undefined,
      agg = undefined,
      y = undefined  // return all columns
    } = this.props;

    if (!dataset || !x || !agg) { return null; }
    const query = JSON.stringify(this.props.getquery());
    return `http://localhost:3000/api/${dataset}/groupby/${x}/${agg}?columns=${y}&query_dict=${query}`;
  }

  async _updateData() {
    const url =
      this._generateApiUrl();  // `/uber/tracts/groupby/sourceid/mean?columns=travel_time`;
    if (!url) { return null; }
    const table = await tableFromIPC(fetch(url, { method: 'GET' }));
    return table.toArray();
  }

  _getOptions(data) {
    console.log(data);
    this.setState({ 'x_axis_indices': data.reduce((a, c) => { return [...a, c[this.props.x]] }, []) })
    console.log(this.state.x_axis_indices);
    return {
      xAxis: {
        type: 'category',
        data: this.props.xaxisdata ? this.props.xaxisdata : this.state.x_axis_indices,
        axisLabel: { color: 'white' },
        splitLine: { show: false },
        name: 'Trips per ' + this.props.x,
        nameLocation: 'middle',
        nameGap: 50,
        datasetIndex: 0
      },
      brush: {
        id: this.props.x,
        toolbox: ['lineX', 'clear'],
        throttleType: 'debounce',
        throttleDelay: 300,
        xAxisIndex: Object.keys(data)
      },
      yAxis: {
        type: 'value',
        axisLabel: { formatter: d3.format('.2s'), color: 'white' },
        splitLine: { show: false },
      },
      series: [{
        type: 'bar',
        id: this.props.x,
        data: data.reduce((a, c) => { return [...a, c[this.props.y]] }, [])
      }],
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
    };
  }

  _onBrushSelected(params) {
    if (params.batch.length > 0 && params.batch[0].areas.length > 0) {
      let selected_range = params.batch[0].areas[0].coordRange.map(r => {
        let idx = (parseInt(r) >= 0) ? parseInt(r) : 0;
        idx = (idx < this.state.x_axis_indices.length) ? idx : this.state.x_axis_indices.length - 1;
        return this.state.x_axis_indices[idx];
      });

      this.props.updatequery({ [this.props.x]: selected_range });
      this.setState({ selected_range: selected_range });
    }
  }

  _onBrushReset(params) {
    if (!params || params.command == 'clear') {
      this.props.updatequery({ [this.props.x]: undefined });
      this.setState({ selected_range: false });
    }
  }

  render() {
    return (
      <Card border='light' bg='secondary' text=
        'light' className={this.props.className + ' text-center'}>
        <Card.Header className='h5'>Trips per{this.props.x}</Card.Header>
        <Card.Body>
          <ReactECharts
            option={this.state.options}
            lazyUpdate={true}
            onEvents={{
              'brush': this._onBrushReset,
              'brushSelected': this._onBrushSelected
            }}
            ref={(e) => { this.echartRef = e; }}
          />
        </Card.Body>
      </Card>)
  }
}
