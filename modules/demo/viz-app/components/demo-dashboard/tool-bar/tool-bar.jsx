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
import styles from './tool-bar.module.css'

export default class ToolBar extends React.Component {
  constructor(props) {
    super(props)
    this.state = {
      selectedTool: 'none',
    }
    this.selectTool = this.selectTool.bind(this);
  }

  selectTool(tool) {
    this.setState(() => ({
      selectedTool: tool
    }));
    this.props.onToolSelect(tool);
  }

  createTool(name, selectedTool) {
    return (
      <div onClick={() => { this.selectTool(name) }} className={`${styles.tool} ${selectedTool == name ? styles.selected : ''}`}>
        {name}
      </div>
    );
  }

  createButton(name, onClick) {
    return <div onClick={onClick} className={styles.tool}>{name}</div>
  }

  render() {
    return (
      <div className={styles.toolBar}>
        {this.createTool('box', this.state.selectedTool)}
        {this.createTool('poly', this.state.selectedTool)}
        {this.createTool('node', this.state.selectedTool)}
        {this.createButton('reset', this.props.onResetClick)}
        {this.createButton('clear', this.props.onClearClick)}
      </div>
    );
  }
}
