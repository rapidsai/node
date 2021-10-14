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

  columns() {
    return [
      {
        Header: 'Index',
        accessor: 'index',
      },
      {
        Header: 'Col Name',
        accessor: 'colname1',
      },
      {
        Header: 'Col Name',
        accessor: 'colname2',
      },
      {
        Header: 'Col Name',
        accessor: 'colname3',
      }
      ,
      {
        Header: 'Col Name',
        accessor: 'colname4',
      }
    ];
  }

  fakeData(i) {
    return {
      index: `testvalue${i}`,
      colname1: `colname${i}`,
      colname2: `colname${i}`,
      colname3: `colname${i}`,
      colname4: `colname${i}`,
      colname2: `colname${i}`,
      colname3: `colname${i}`,
      colname4: `colname${i}`,
    };
  }

  dataTable() {
    return (
      <Tabs>
        <TabList>
          <Tab>Node List</Tab>
          <Tab>Edge List</Tab>
        </TabList>

        <TabPanel>
          <ExtendedTable
            cols={this.columns()}
            data={[
              this.fakeData(0), this.fakeData(1), this.fakeData(2), this.fakeData(3), this.fakeData(4), this.fakeData(5),
              this.fakeData(6), this.fakeData(7), this.fakeData(8), this.fakeData(9), this.fakeData(10), this.fakeData(11),
              this.fakeData(12)
            ]}
          />
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
