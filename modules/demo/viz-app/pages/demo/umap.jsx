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

import { Tab, Tabs, TabList, TabPanel } from 'react-tabs';
import React from 'react';
import DemoDashboard from "../../components/demo-dashboard/demo-dashboard";
import HeaderUnderline from '../../components/demo-dashboard/header-underline/header-underline';
import ExtendedTable from '../../components/demo-dashboard/extended-table/extended-table';

export default class UMAP extends React.Component {
  constructor(props) {
    super(props);
    this.dataTable = this.dataTable.bind(this);
    this.dataMetrics = this.dataMetrics.bind(this);
  }

  demoView() {
    return (
      <p>Demo goes here.</p>
    );
  }

  dataTable() {
    return (
      <Tabs>
        <TabList>
          <Tab>Node List</Tab>
          <Tab>Edge List</Tab>
        </TabList>

        <TabPanel>
          <ExtendedTable />
        </TabPanel>
        <TabPanel>
          <div>This is edge list</div>
        </TabPanel>
      </Tabs>
    )
  }

  dataMetrics() {
    return (
      <div style={{ padding: 10, color: 'white' }}>
        <HeaderUnderline title={"Data Metrics"} fontSize={18} color={"white"}>
          <div>{'>'} 100,001,203 Edges</div>
          <div>{'>'} 20,001,525 Nodes</div>
          <div>{'>'} 5.2GB</div>
        </HeaderUnderline>
      </div>
    )
  }

  render() {
    return (
      <DemoDashboard demoName={"UMAP Demo"}
        demoView={this.demoView()}
        onLoadClick={(fileName) => { console.log(fileName) }}
        onRenderClick={() => { console.log("Render Clicked") }}
        dataTable={this.dataTable()}
        dataMetrics={this.dataMetrics()}
      />
    )
  }
}
