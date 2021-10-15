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

import { io } from "socket.io-client";
import SimplePeer from "simple-peer";
import wrtc from 'wrtc';

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

export default class UMAP extends React.Component {
  constructor(props) {
    super(props);
    this.dataTable = this.dataTable.bind(this);
    this.dataMetrics = this.dataMetrics.bind(this);
    this.videoRef = React.createRef();
    this.peer = null;
  }

  componentDidMount() {
    this.socket = io("localhost:8080", { transports: ['websocket'], reconnection: false, query: { width: 2000, height: 500 } });
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

    this.dispatchRemoteEvent(this.videoRef.current, 'blur');
    this.dispatchRemoteEvent(this.videoRef.current, 'wheel');
    this.dispatchRemoteEvent(this.videoRef.current, 'mouseup');
    this.dispatchRemoteEvent(this.videoRef.current, 'mousemove');
    this.dispatchRemoteEvent(this.videoRef.current, 'mousedown');
    this.dispatchRemoteEvent(this.videoRef.current, 'mouseenter');
    this.dispatchRemoteEvent(this.videoRef.current, 'mouseleave');
    // this.dispatchRemoteEvent(window, 'beforeunload');
    // this.dispatchRemoteEvent(document, 'keydown');
    // this.dispatchRemoteEvent(document, 'keypress');

  }

  dispatchRemoteEvent(target, type) {
    let timeout = null;
    target.addEventListener(type, (e) => {
      if (e.target === this.videoRef.current) {
        e.preventDefault();
      }
      if (!timeout) {
        timeout = setTimeout(() => { timeout = null; }, 1000 / 60);
        if (this.peer) {
          this.peer.send(JSON.stringify({ type: 'event', data: serializeEvent(e) }));
        }
      }
    });
  }


  demoView() {
    return (
      <video autoPlay muted width="2000" height="500"
        ref={this.videoRef}
      // onMouseUp={remoteEvent}
      // onMouseDown={remoteEvent} onMouseMove={remoteEvent}
      // onMouseLeave={remoteEvent} onWheel={remoteEvent}
      // onMouseEnter={remoteEvent} onBlur={remoteEvent}
      > Demo goes here</video>
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
            data={[this.fakeData(0), this.fakeData(1), this.fakeData(2), this.fakeData(3), this.fakeData(4), this.fakeData(5), this.fakeData(6), this.fakeData(7)]}
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
      <DemoDashboard demoName={"Graph Demo"}
        demoView={this.demoView()}
        onLoadClick={(fileName) => { console.log(fileName) }}
        onRenderClick={() => { console.log("Render Clicked") }}
        dataTable={this.dataTable()}
        dataMetrics={this.dataMetrics()}
      />
    )
  }
}
