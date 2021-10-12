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
import Layout from './layout/layout';
import TempDemoView from './temp-demo-view/temp-demo-view';
import DataRow from './data-row/data-row';
import Footer from './footer/footer';
import { Container } from 'react-bootstrap';
import SlideMenu from './slide-menu/slide-menu';

export default class DemoDashboard extends React.Component {
  render() {
    const { demoName } = this.props;

    return (
      <div id="outer-container">
        <SlideMenu
          onLoadClick={this.props.onLoadClick}
          onRenderClick={this.props.onRenderClick}
        />
        <Layout id="page-wrap" title={demoName}>
          <TempDemoView />
          <Container fluid style={{ paddingTop: 20 }}>
            <DataRow dataTable={this.props.dataTable} dataMetrics={this.props.dataMetrics} />
            <Footer />
          </Container>
        </Layout>
      </div>
    )
  }
}
