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

const SimplePeer = require('simple-peer');

import isBrowser from '../is-browser';
import React, { Component, createRef, forwardRef, RefObject } from 'react';

type Props = {
  rtcId: string;
  render: (props: any, ...children: React.ReactNode[]) => React.ReactElement;
};
type State = { stream: MediaSource };

const defaultProps = { muted: true, autoPlay: true, width: 800, height: 600 };

export default class WebRTCFrame extends Component<Props, State> {

  private _peer: any;
  private _videoRef = createRef<HTMLVideoElement>();
  private _onKeyEvent = this._eventHandler(serializeKeyEvent);
  private _onFocusEvent = this._eventHandler(serializeFocusEvent);
  private _onMouseEvent = this._eventHandler(serializeMouseEvent);
  private _onWheelEvent = this._eventHandler(serializeWheelEvent);

  componentDidMount() {
    if (isBrowser) {
      this._peer = new SimplePeer({ initiator: true, sdpTransform })
        .on('stream', play.bind(null, this._videoRef))
        .on('connect', () => console.log('peer connected'))
        .on('close', () => console.log(`peer ${this.props.rtcId} close`))
        .on('error', () => console.log(`peer ${this.props.rtcId} error`));
      negotiate(this.props.rtcId, this._peer);
      window.addEventListener('beforeunload', this._destroyPeer.bind(this));
    }
  }

  componentWillUnmount() {
    this._destroyPeer();
  }

  render() {
    const { render, children, ...props } = this.props;
    return render({
      ...defaultProps, ...props,
      ...this._eventHandlers(),
      ref: this._videoRef
    }, children);
  }

  protected _destroyPeer() {
    if (isBrowser && this._peer && this._peer.connected) {
      this._peer.destroy();
      this._peer = null;
    }
  }

  protected _eventHandler(serialize: (event: any) => any) {
    return () => {
      if (this._peer && this._peer.connected) {
        this._peer.send(JSON.stringify({
          type: 'event',
          data: serialize(event)
        }) + '\n');
      }
    }
  }

  protected _eventHandlers() {
    return {
      onKeyDown: this._onKeyEvent,
      onKeyPress: this._onKeyEvent,
      onKeyUp: this._onKeyEvent,
      onFocus: this._onFocusEvent,
      onBlur: this._onFocusEvent,
      onClick: this._onMouseEvent,
      onContextMenu: this._onMouseEvent,
      onDoubleClick: this._onMouseEvent,
      onDrag: this._onMouseEvent,
      onDragEnd: this._onMouseEvent,
      onDragEnter: this._onMouseEvent,
      onDragExit: this._onMouseEvent,
      onDragLeave: this._onMouseEvent,
      onDragOver: this._onMouseEvent,
      onDragStart: this._onMouseEvent,
      onDrop: this._onMouseEvent,
      onMouseDown: this._onMouseEvent,
      onMouseEnter: this._onMouseEvent,
      onMouseLeave: this._onMouseEvent,
      onMouseMove: this._onMouseEvent,
      onMouseOut: this._onMouseEvent,
      onMouseOver: this._onMouseEvent,
      onMouseUp: this._onMouseEvent,
      onWheel: this._onWheelEvent,
    };
  }
}

function negotiate(rtcId: string, peer: any) {
  return peer.once('signal', async (offer: any) => {
    try {
      console.log(`peer ${rtcId} offer:`, offer);
      const response = await fetch('/api/rtc/signal', {
        method: 'POST',
        body: JSON.stringify({ rtcId, offer }),
        headers: { 'Content-Type': 'application/json' },
      });
      if (response.ok) {
        const answer = await response.json();
        console.log(`peer ${rtcId} got answer:`, answer);
        negotiate(rtcId, peer).signal(answer);
      } else {
        console.log(`peer ${rtcId} bad answer`);
        peer.destroy();
      }
    } catch (e) {
      console.error(e);
      peer.destroy();
    }
  });
}

function play({ current: video }: RefObject<HTMLVideoElement>, source: MediaSource) {
  if (video) {
    if ('srcObject' in video) {
      video.srcObject = source;
    } else { // for older browsers
      (video as any).src = window.URL.createObjectURL(source);
    }
  }
  return video?.play();
}

function sdpTransform(sdp: string) {

  // Remove bandwidth restrictions
  // https://github.com/webrtc/samples/blob/89f17a83ed299ef28d45b933419d809b93d41759/src/content/peerconnection/bandwidth/js/main.js#L240
  sdp = sdp.replace(/b=AS:.*\r\n/, '').replace(/b=TIAS:.*\r\n/, '');
  // Force h264 encoding by removing VP8/9 codecs from the sdp
  sdp = onlyH264(sdp);

  return sdp;

  function onlyH264(sdp: string) {
    // remove non-h264 codecs from the supported codecs list
    const videos = sdp.match(/^m=video.*$/gm);
    if (videos) {
      return videos.map((video) =>
        [
          video,
          [
            ...getCodecIds(sdp, 'VP9'),
            ...getCodecIds(sdp, 'VP8'),
            ...getCodecIds(sdp, 'HEVC'),
            ...getCodecIds(sdp, 'H265')
          ]
        ] as [string, string[]]
      ).reduce((sdp, [video, ids]) => ids.reduce((sdp, id) => [
        new RegExp(`^a=fmtp:${id}(.*?)$`, 'gm'),
        new RegExp(`^a=rtpmap:${id}(.*?)$`, 'gm'),
        new RegExp(`^a=rtcp-fb:${id}(.*?)$`, 'gm'),
      ].reduce((sdp, expr) => sdp.replace(expr, ''), sdp), sdp)
        .replace(video, ids.reduce((video, id) => video.replace(` ${id}`, ''), video)), sdp)
        .replace('\r\n', '\n').split('\n').map((x) => x.trim()).filter(Boolean).join('\r\n') + '\r\n';
    }

    return sdp;
  }

  function getCodecIds(sdp: string, codec: string) {
    return getIdsForMatcher(sdp, new RegExp(
      `^a=rtpmap:(?<id>\\d+)\\s+${codec}\\/\\d+$`, 'm'
    )).reduce((ids, id) => [
      ...ids, id, ...getIdsForMatcher(sdp, new RegExp(
        `^a=fmtp:(?<id>\\d+)\\s+apt=${id}$`, 'm'
      ))
    ], [] as string[]);
  }

  function getIdsForMatcher(sdp: string, matcher: RegExp) {
    const ids = [];
    /** @type RegExpMatchArray */
    let res, str = '' + sdp, pos = 0;
    for (; res = str.match(matcher); str = str.slice(pos)) {
      pos = res.index! + res[0].length;
      if (res.groups) { ids.push(res.groups.id); }
    }
    return ids;
  }
}

function serializeKeyEvent(event: React.KeyboardEvent) {
  return {
    type: event.type,
    altKey: event.altKey,
    charCode: event.charCode,
    ctrlKey: event.ctrlKey,
    key: event.key,
    keyCode: event.keyCode,
    locale: event.locale,
    location: event.location,
    metaKey: event.metaKey,
    repeat: event.repeat,
    shiftKey: event.shiftKey,
    which: event.which,
  };
}

function serializeFocusEvent(event: React.FocusEvent) {
  return { type: event.type };
}

function serializeMouseEvent(event: React.MouseEvent) {
  return {
    type: event.type,
    altKey: event.altKey,
    button: event.button,
    buttons: event.buttons,
    clientX: event.clientX,
    clientY: event.clientY,
    ctrlKey: event.ctrlKey,
    metaKey: event.metaKey,
    pageX: event.pageX,
    pageY: event.pageY,
    screenX: event.screenX,
    screenY: event.screenY,
    shiftKey: event.shiftKey,
  };
}

function serializeWheelEvent(event: React.WheelEvent) {
  return {
    type: event.type,
    deltaX: event.deltaX,
    deltaY: event.deltaY,
    deltaZ: event.deltaZ,
    deltaMode: event.deltaMode,
  };
}
