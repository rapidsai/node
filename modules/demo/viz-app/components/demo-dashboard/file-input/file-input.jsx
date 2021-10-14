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

export default class FileInput extends React.Component {
  constructor(props) {
    super(props)
    this.uploadFile = this.uploadFile.bind(this);
  }

  uploadFile(event) {
    let file = event.target.files[0];
    this.props.onChange(file.name);
  }

  render() {
    return <label style={{ width: 120, height: 0 }}>
      <p className={"textButton"}>Select Data ▼</p>
      <input type="file" style={{ display: "none" }} onChange={this.uploadFile} />
    </label>
  }
}
