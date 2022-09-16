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
import DemoDashboard from "../../components/demo-dashboard/demo-dashboard";

import { io } from "socket.io-client";
import SimplePeer from "simple-peer";

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

export default class PointCloud extends React.Component {

  constructor(props) {
    super(props);
    this.videoRef = React.createRef();
    this.peerConnected = false;
  }

  componentDidMount() {
    console.log(this.videoRef.current.width);
    this.socket = io("localhost:8080", { transports: ['websocket'], reconnection: true, query: { width: this.videoRef.current.width, height: this.videoRef.current.height } });
    this.peer = new SimplePeer({
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
        />
      </div>
    );
  }

  render() {
    return (
      <DemoDashboard demoName={"Point Cloud Demo"}
        demoView={this.demoView()}
        onLoadClick={(fileName) => { console.log(fileName) }}
        onRenderClick={() => { console.log("Render Clicked") }}
      />
    )
  }
}
