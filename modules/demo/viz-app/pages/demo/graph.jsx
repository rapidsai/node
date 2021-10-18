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
import { Form, Col, Row } from 'react-bootstrap';
import React from 'react';
import DemoDashboard from "../../components/demo-dashboard/demo-dashboard";
import HeaderUnderline from '../../components/demo-dashboard/header-underline/header-underline';
import ExtendedTable from '../../components/demo-dashboard/extended-table/extended-table';
import ToolBar from '../../components/demo-dashboard/tool-bar/tool-bar';

import { io } from "socket.io-client";
import SimplePeer from "simple-peer";
import wrtc from 'wrtc';
import axios from 'axios';

function onlyH264(sdp) {
  // remove non-h264 codecs from the supported codecs list
  const videos = sdp.match(/^m=video.*$/gm);
  if (videos) {
    return videos.map((video) => [video, [
      ...getCodecIds(sdp, 'VP9'),
      ...getCodecIds(sdp, 'VP8'),
      ...getCodecIds(sdp, 'HEVC'),
      ...getCodecIds(sdp, 'H265')
    ]]).reduce((sdp, [video, ids]) => ids.reduce((sdp, id) => [
      new RegExp(`^a=fmtp:${id}(.*?)$`, 'gm'),
      new RegExp(`^a=rtpmap:${id}(.*?)$`, 'gm'),
      new RegExp(`^a=rtcp-fb:${id}(.*?)$`, 'gm'),
    ].reduce((sdp, expr) => sdp.replace(expr, ''), sdp), sdp)
      .replace(video, ids.reduce((video, id) => video.replace(` ${id}`, ''), video)), sdp)
      .replace('\r\n', '\n').split('\n').map((x) => x.trim()).filter(Boolean).join('\r\n') + '\r\n';
  }

  return sdp;

  function getCodecIds(sdp, codec) {
    return getIdsForMatcher(sdp, new RegExp(
      `^a=rtpmap:(?<id>\\d+)\\s+${codec}\\/\\d+$`, 'm'
    )).reduce((ids, id) => [
      ...ids, id, ...getIdsForMatcher(sdp, new RegExp(
        `^a=fmtp:(?<id>\\d+)\\s+apt=${id}$`, 'm'
      ))
    ], []);
  }

  function getIdsForMatcher(sdp, matcher) {
    const ids = [];
    /** @type RegExpMatchArray */
    let res, str = '' + sdp, pos = 0;
    for (; res = str.match(matcher); str = str.slice(pos)) {
      pos = res.index + res[0].length;
      if (res.groups) { ids.push(res.groups.id); }
    }
    return ids;
  }
}


function serializeEvent(original) {
  return Object
    .getOwnPropertyNames(Object.getPrototypeOf(original))
    .reduce((serialized, field) => {
      switch (typeof original[field]) {
        case 'object':
        case 'symbol':
        case 'function': break;
        default: serialized[field] = original[field];
      }
      serialized = { ...serialized, x: serialized.layerX, y: serialized.layerY };
      return serialized;
    }, { type: original.type });
}

export default class Graph extends React.Component {

  constructor(props) {
    super(props);
    this.dataTable = this.dataTable.bind(this);
    this.dataMetrics = this.dataMetrics.bind(this);
    this.videoRef = React.createRef();
    this.peerConnected = false;
    this.state = {
      nodes: {
        tableData: [{}],
        tableColumns: [{ Header: "Loading...", accessor: "Loading..." }],
        length: 0
      },
      edges: {
        tableData: [{}],
        tableColumns: [{ Header: "Loading...", accessor: "Loading..." }],
        length: 0
      },
      fileUploadStatus: {},
      nodesFileOptions: [],
      edgesFileOptions: [],
      nodesFile: "",
      edgesFile: "",
      nodesParams: [],
      edgesParams: [],
      nodesRenderColumns: {
        x: "x",
        y: "y",
        color: "",
        size: "",
        id: ""
      },
      edgesRenderColumns: {
        src: "src", dst: "dst",
        color: "", bundle: "", id: ""
      },
      gpuLoadStatus: "not loaded",
      forceAtlas2: false
    }
  }

  componentDidMount() {
    console.log(this.videoRef.current.width);
    this.socket = io("localhost:8080", {
      transports: ['websocket'], reconnection: true,
      query: { width: this.videoRef.current.width, height: this.videoRef.current.height, layout: this.state.forceAtlas2 }
    });
    this.peer = new SimplePeer({
      wrtc: wrtc,
      trickle: true,
      initiator: true,
      sdpTransform: (sdp) => {
        // Remove bandwidth restrictions
        // https://github.com/webrtc/samples/blob/89f17a83ed299ef28d45b933419d809b93d41759/src/content/peerconnection/bandwidth/js/main.js#L240
        sdp = sdp.replace(/b=AS:.*\r\n/, '').replace(/b=TIAS:.*\r\n/, '');
        // Force h264 encoding by removing all VP8/9 codecs from the sdp
        sdp = onlyH264(sdp);
        return sdp;
      }
    });

    // Negotiate handshake
    this.socket.on('signal', (data) => this.peer.signal(data));
    this.peer.on('signal', (data) => this.socket.emit('signal', data));
    this.peer.on('stream', (stream) => {
      this.videoRef.current.srcObject = stream;
    });
    this.peer.on('connect', () => {
      this.peerConnected = true;
    });

    function processColumns(columnObject) {
      return Object.keys(columnObject).reduce(
        (prev, curr) => {
          return prev.concat([{ "Header": curr, "accessor": curr }]);
        }, []);
    }

    this.peer.on('data', (data) => {
      const decodedjson = JSON.parse(new TextDecoder().decode(data));
      const keys = Object.keys(decodedjson.data);
      keys.forEach((key) => {
        if (['edges', 'nodes'].includes(key) && decodedjson.data[[key]].data.length > 0) {
          this.setState({
            [key]: {
              tableColumns: processColumns(decodedjson.data[[key]].data[0]),
              tableData: decodedjson.data[[key]].data,
              length: decodedjson.data[[key]].length
            }
          });
        }
      });


    });

    this.dispatchRemoteEvent(this.videoRef.current, 'blur');
    this.dispatchRemoteEvent(this.videoRef.current, 'wheel');
    this.dispatchRemoteEvent(this.videoRef.current, 'mouseup');
    this.dispatchRemoteEvent(this.videoRef.current, 'mousemove');
    this.dispatchRemoteEvent(this.videoRef.current, 'mousedown');
    this.dispatchRemoteEvent(this.videoRef.current, 'mouseenter');
    this.dispatchRemoteEvent(this.videoRef.current, 'mouseleave');
  }

  dispatchRemoteEvent(target, type) {
    let timeout = null;
    target.addEventListener(type, (e) => {
      if (e.target === this.videoRef.current) {
        e.preventDefault();
      }
      if (!timeout) {
        timeout = setTimeout(() => { timeout = null; }, 1000 / 60);
        if (this.peerConnected) {
          this.peer.send(JSON.stringify({ type: 'event', data: serializeEvent(e) }));
        }
      }
    });
  }



  demoView() {
    return (
      <div style={{ width: "2000px", height: "500px", display: "flex" }}>
        <video autoPlay muted width="1860px" height="500px" style={{ flexDirection: "row", float: "left" }}
          ref={this.videoRef}
        /**
         *  Using synthetic react events is throwing:
         * `Unable to preventDefault inside passive event listener due to target being treated as passive`,
         *  Temporarily using dom event listeners
        */
        // onMouseUp={remoteEvent}
        // onMouseDown={remoteEvent} onMouseMove={remoteEvent}
        // onMouseLeave={remoteEvent} onWheel={remoteEvent}
        // onMouseEnter={remoteEvent} onBlur={remoteEvent}
        />
        <ToolBar style={{ flexDirection: "row", width: "200px" }}
          onResetClick={() => { console.log("reset") }}
          onClearClick={() => { this.peer.send(JSON.stringify({ type: 'clearSelections', data: true })); }}
          onToolSelect={(tool) => { this.peer.send(JSON.stringify({ type: 'pickingMode', data: tool })); }}
        />
      </div>
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
          <ExtendedTable
            cols={this.state.nodes.tableColumns}
            data={this.state.nodes.tableData}
          />
        </TabPanel>
        <TabPanel>
          <ExtendedTable
            cols={this.state.edges.tableColumns}
            data={this.state.edges.tableData}
          />
        </TabPanel>
      </Tabs>
    )
  }

  dataMetrics() {
    return (
      <div style={{ padding: 10, color: 'white' }}>
        <HeaderUnderline title={"Data Metrics"} fontSize={18} color={"white"}>
          <div>{'>'} {this.state.nodes.length} Edges</div>
          <div>{'>'} {this.state.edges.length} Nodes</div>
        </HeaderUnderline>
      </div>
    )
  }

  uploadFile(file) {
    console.log("uploading", file.name, this);
    let formData = new FormData();
    formData.append("file", file);
    axios.post("http://localhost:8080/uploadFile", formData, {
    }).then(respone => { console.log(respone.statusText); })
      .then(success => {
        console.log("file uploaded successfully");
      })
      .catch(error => {
        console.log("error");
      });
  }

  reloadFiles() {
    axios.get("http://localhost:8080/getFileNames?id=" + this.socket.id).then((response) => {
      this.setState({
        nodesFileOptions: response.data,
        edgesFileOptions: response.data,
        nodesFile: response.data[0],
        edgesFile: response.data[0]
      })
    }).catch((error) => {
      // handle error
      console.log(error);
    }).then(() => {
      console.log("request complete")
    })
  }

  fetchDFParameters() {
    axios.get("http://localhost:8080/fetchDFParameters", {
      params: {
        id: this.socket.id
      }
    }).then((response) => {
      this.setState({
        nodesParams: response.data.nodesParams,
        edgesParams: response.data.edgesParams,
        nodesRenderColumns: {
          x: response.data.nodesParams[0],
          y: response.data.nodesParams[0],
          color: response.data.nodesParams[0],
          size: response.data.nodesParams[0],
          id: response.data.nodesParams[0]
        },
        edgesRenderColumns: {
          src: response.data.edgesParams[0], dst: response.data.edgesParams[0],
          color: response.data.edgesParams[0], bundle: response.data.edgesParams[0], id: response.data.edgesParams[0]
        }
      });
    }).catch((error) => { }).then(() => { });
  }

  loadOnGPU() {
    axios.get("http://localhost:8080/loadOnGPU", {
      params: {
        id: this.socket.id,
        nodes: this.state.nodesFile,
        edges: this.state.edgesFile
      }
    }).then((response) => {
      this.setState({
        gpuLoadStatus: response.data
      });
    }).catch((error) => {
      this.setState({
        gpuLoadStatus: "not loaded"
      })
    }).then(() => {
      this.fetchDFParameters()
    });
  }

  updateRenderColumns(df, param, value) {
    if (df == "nodes") {
      this.setState({
        nodesRenderColumns: Object.assign(this.state.nodesRenderColumns, { [param]: value })
      });
    } else {
      this.setState({
        edgesRenderColumns: Object.assign(this.state.edgesRenderColumns, { [param]: value })
      });
    }
  }

  getDFParameters() {
    if (this.state.gpuLoadStatus !== "not loaded") {
      return (
        <Form.Group>
          <Form.Label>Nodes</Form.Label>
          <Form.Row>
            {["x", "y", "color", "size", "id"].map((param) => <Form.Group as={Col} md="2">
              <Form.Label>{param}</Form.Label>
              <Form.Control as="select" custom onChange={(e) => { this.updateRenderColumns("nodes", param, e.target.value); }}>
                {this.state.nodesParams.map((obj) => <option value={obj}>{obj}</option>)}
              </Form.Control>
            </Form.Group>)}
          </Form.Row>
          <Form.Label>Edges</Form.Label>
          <Form.Row>
            {["src", "dst", "color", "bundle", "id"].map((param) => <Form.Group as={Col} md="2">
              <Form.Label>{param}</Form.Label>
              <Form.Control as="select" custom onChange={(e) => { this.updateRenderColumns("edges", param, e.target.value); }}>
                {this.state.edgesParams.map((obj) => <option value={obj}>{obj}</option>)}
              </Form.Control>
            </Form.Group>)}
          </Form.Row>
        </Form.Group>
      )
    } else {
      return (<></>)
    }
  }

  getCustomComponents() {
    return (<div>
      <p className={"textButton"} onClick={() => this.reloadFiles()}>[Refresh files]</p>
      <Form.Group>
        <Form.Label>Select Nodes</Form.Label>
        <Form.Control as="select" custom onChange={(e) => { this.setState({ nodesFile: e.target.value }); }}>
          {this.state.nodesFileOptions.map((obj) => <option value={obj}>{obj}</option>)}
        </Form.Control>
        <Form.Label>Select Edges</Form.Label>
        <Form.Control as="select" custom onChange={(e) => { this.setState({ edgesFile: e.target.value }); }}>
          {this.state.edgesFileOptions.map((obj) => <option value={obj}>{obj}</option>)}
        </Form.Control>
        <p className={"textButton"} onClick={() => this.loadOnGPU()}>[Load on GPU] {this.state.gpuLoadStatus}</p>
      </Form.Group>
      {this.getDFParameters()}

      <h4 style={{ color: 'black' }}><Form.Check
        type="switch"
        id="custom-switch"
        label="ForceAtlas2"
        onChange={((e) => {
          this.peer.send(JSON.stringify({ type: 'layout', data: e.target.checked }));
        })}
      /> </h4>
    </div >)
  }

  onRenderClick() {
    axios.post("http://localhost:8080/updateRenderColumns", {
      nodes: this.state.nodesRenderColumns,
      edges: this.state.edgesRenderColumns,
      id: this.socket.id
    }).then((response) => {
      console.log("success");
    }).catch((error) => {
      console.log("error");
    })
  }
  render() {
    return (
      <DemoDashboard demoName={"Graph Demo"}
        demoView={this.demoView()}
        onLoadClick={this.uploadFile}
        customComponents={this.getCustomComponents()}
        onRenderClick={() => this.onRenderClick()}
        dataTable={this.dataTable()}
        dataMetrics={this.dataMetrics()}
      />
    )
  }
}
