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

import * as React from 'react';
import styles from './tool-bar.module.css';
import Image from 'next/image';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faVectorSquare, faDrawPolygon, faMousePointer, faSearchMinus, faTimes } from '@fortawesome/free-solid-svg-icons';

export default class ToolBar extends React.Component {
  constructor(props) {
    super(props)
    this.state = {
      selectedTool: 'click',
    }
    this.selectTool = this.selectTool.bind(this);
  }

  selectTool(tool) {
    this.setState(() => ({
      selectedTool: tool
    }));
    this.props.onToolSelect(tool);
  }

  createTool(name, icon, selectedTool) {
    return (
      <div onClick={() => { this.selectTool(name) }} className={`${styles.tool} ${selectedTool == name ? styles.selected : ''}`}>
        <FontAwesomeIcon icon={icon} />
      </div>
    );
  }

  createButton(tag, onClick) {
    return (
      <div onClick={onClick} className={styles.tool}>
        {tag}
      </div>
    )
  }

  render() {
    return (
      <div className={styles.toolBar}>
        {this.createTool('boxSelect', faVectorSquare, this.state.selectedTool)}
        {/* {this.createTool('poly', faDrawPolygon, this.state.selectedTool)} */}
        {this.createTool('click', faMousePointer, this.state.selectedTool)}
        {/* {this.createButton(<Image src="/images/zoom.png" width={20} height={20} />, this.props.onResetClick)} */}
        {this.createButton(<Image src="/images/reset.png" width={20} height={20} />, this.props.onClearClick)}
      </div>
    );
  }
}
