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
import FileInput from '../file-input/file-input';
import HeaderUnderline from '../header-underline/header-underline';
import { slide as Menu } from 'react-burger-menu';
import { Row, Col } from 'react-bootstrap';

export default class SlideMenu extends React.Component {
  constructor(props) {
    super(props)
    this.state = {
      selectedFilePath: "",
    }
    this.onDataChange = this.onDataChange.bind(this);
    this.onLoadClick = this.onLoadClick.bind(this);
    this.onRenderClick = this.onRenderClick.bind(this);
  }

  onDataChange(filePath) {
    this.setState({
      selectedFilePath: filePath
    });
  }

  onLoadClick() {
    this.props.onLoadClick(this.state.selectedFilePath);
  }

  onRenderClick() {
    this.props.onRenderClick();
  }

  render() {
    return (
      <Menu pageWrapId={"page-wrap"} outerContainerId={"outer-container"} width={'50vw'}>
        <HeaderUnderline title={"Data Source"}>
          <Row>
            <Col className={"col-auto"}>
              <FileInput onChange={this.onDataChange}></FileInput>
              <p style={{ color: "black" }}>Selection: {this.state.selectedFilePath}</p>
            </Col>
            <Col className={"max"} ><div className={"d-flex"} /></Col>
            <Col className={"col-auto"}>
              <p className={"textButton"} onClick={this.onLoadClick}>[Load]</p>
            </Col>
          </Row>
        </HeaderUnderline>
        <div style={{ height: 20 }} />
        <HeaderUnderline title={"Visualization"}>
          <p className={"textButton"} onClick={this.onRenderClick}>[Render]</p>
        </HeaderUnderline>
      </Menu >
    );
  }
}
