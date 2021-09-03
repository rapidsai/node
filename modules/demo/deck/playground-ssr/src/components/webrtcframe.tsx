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

import React, { Component, createRef } from 'react';
import { createPeer } from '../../webrtc/client';
import isBrowser from '../is-browser';

type Props = {
  json?: any;
  width?: number;
  height?: number;
};

const defaultProps = { muted: true, autoPlay: true, width: 800, height: 600 };

export default class WebRTCFrame extends Component<Props> {

  private _peer: any;
  private _sock: any;
  private _videoRef = createRef<HTMLVideoElement>();
  private _onKeyEvent = this._eventHandler(serializeKeyEvent);
  private _onFocusEvent = this._eventHandler(serializeFocusEvent);
  private _onMouseEvent = this._eventHandler(serializeMouseEvent);
  private _onWheelEvent = this._eventHandler(serializeWheelEvent);

  componentDidMount() {
    if (isBrowser) {
      const { peer, sock } = createPeer();
      this._peer = peer;
      this._sock = sock;
      this._play = this._play.bind(this);
      this._peer.on('stream', this._play);
      this._destroy = this._destroy.bind(this)
      window.addEventListener('beforeunload', this._destroy);
    }
  }

  componentWillUnmount() {
    window.removeEventListener('beforeunload', this._destroy);
    this._destroy();
  }

  render() {
    const { json, width, height } = this.props;
    this._send('render', { json, width, height });
    return (
      <video
        {...defaultProps}
        width={width}
        height={height}
        {...this._eventHandlers()}
        ref={this._videoRef}
      />
    );
  }

  protected _send(type: string, data: any = {}) {
    if (isBrowser && this._peer && this._peer.connected) {
      this._peer.send(JSON.stringify({ type, data }));
    }
  }

  protected _play(source: MediaSource) {
    const video = this._videoRef.current;
    console.log('playing video:', source);
    if (video) {
      if ('srcObject' in video) {
        video.srcObject = source;
      } else {  // for older browsers
        (video as any).src = window.URL.createObjectURL(source);
      }
      const { json, width, height } = this.props;
      Array.from({ length: 5 }, (_, i) => i).forEach(() => {
        this._send('render', { json, width, height });
      });
    }
    return video?.play();
  }

  protected _destroy() {
    if (isBrowser) {
      if (this._peer && this._peer.connected) {
        this._peer.destroy();
        this._peer = null;
      }
      if (this._sock && this._sock.connected) {
        this._sock.destroy();
        this._sock = null;
      }
    }
  }

  protected _eventHandler(serialize: (event: any) => any) {
    return (event: any) => {
      this._send('event', serialize(event));
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
