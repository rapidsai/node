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
      selectedFile: {}
    }
    this.onDataChange = this.onDataChange.bind(this);
    this.onLoadClick = this.onLoadClick.bind(this);
    this.onRenderClick = this.onRenderClick.bind(this);
  }

  onDataChange(file) {
    this.setState({
      selectedFile: file,
    });
  }

  onLoadClick() {
    console.log("hereherhehre");
    this.props.onLoadClick(this.state.selectedFile);
  }

  onRenderClick() {
    this.props.onRenderClick();
  }


  render() {
    return (
      <Menu pageWrapId={"page-wrap"} outerContainerId={"outer-container"} width={'50vw'}>
        <HeaderUnderline title={"Data Source"} color={"white"}>
          <Row>
            <Col className={"col-auto"}>
              <FileInput onChange={this.onDataChange} useWhite={true}>
                Select Data ▼
              </FileInput>
            </Col>
            <p style={{ color: "black" }}>Selection: {this.state.selectedFile.name}</p>
            <Col className={"max"} ><div className={"d-flex"} /></Col>
            <Col className={"col-auto"}>
              <p className={"whiteTextButton"} onClick={this.onLoadClick}>[upload]</p>
            </Col>
          </Row>
        </HeaderUnderline>
        <div style={{ height: 20 }} />
        <HeaderUnderline title={"Visualization"} color={"white"}>
          <p className={"whiteTextButton"} onClick={this.onRenderClick}>[Render]</p>
        </HeaderUnderline>
      </Menu >
    );
  }
}
