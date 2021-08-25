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

import {nanoid} from 'nanoid';
import React from 'react';
import isBrowser from '../is-browser';

type Props = {
  rtcId: string;
  width?: number,
  height?: number,
  muted?: boolean,
  autoPlay?: boolean,
};

let render: (props: Props, ...children: React.ReactNode[]) => React.ReactElement;

function renderVideo({rtcId, ...props}: Props) { return React.createElement('video', props); }

if (isBrowser) {
  render = renderVideo;
} else {
  const numWorkers = 1;
  const workers    = ((workers_: import('child_process').ChildProcess[] = []) => {
    return () => {
      if (workers_.length === 0) {
        const {fork} = require('child_process');
        for (let i = -1; ++i < numWorkers;) {
          workers_[i] = fork(`${__dirname}/worker.js`, {
            serialization: 'advanced',
            stdio: ['pipe', 'inherit', 'inherit', 'ipc'],
            execArgv: ['-r', 'esm'],
            env: {
              ...process.env,
              DISPLAY: undefined,
              WAYLAND_DISPLAY: undefined,
            },
          });
        }
      }
      return workers_;
    }
  })();

  const renderRemote = (props: Props, ...children: React.ReactNode[]) => {
    process.on('beforeExit', (code) => {
      workers().forEach((worker) => {
        if (worker && !worker.killed) {
          worker.send({type: 'exit'});
          worker.kill(code || 'SIGKILL');
        }
      });
    });

    const wrtc           = require('wrtc');
    const {getPeer}      = require('./broker');
    const ReactDOMServer = require('react-dom/server');

    const MediaStream    = wrtc.MediaStream;
    const RTCVideoSource = wrtc.nonstandard.RTCVideoSource;

    let peer = getPeer(props.rtcId);
    let uids = new Array<string>();
    let stream: any, source: any;

    peer.once('connect', onConnect);
    peer.once('close', onClose);
    peer.once('error', onClose);

    return renderVideo(props);

    function onClose() {
      workers()  //
        .filter((wkr) => hasListener(wkr, 'message', onRender))
        .forEach((wkr) => { wkr.off('message', onRender); });

      if (peer) {
        peer.removeListener('data', onUpdate);
        if (stream) {
          peer.removeStream(stream);
          source = null;
          stream = null;
        }
        peer = null;
      }
    }

    function onConnect() {
      if (peer && peer.connected) {
        peer.on('data', onUpdate);
        const id = `${props.rtcId}:video`;
        stream   = new MediaStream({id, isScreencast: true});
        source   = new RTCVideoSource({isScreencast: true});
        stream.addTrack(source.createTrack());
        peer.addStream(stream);
      }
    }

    function onUpdate(data?: any) {
      if (peer && peer.connected) {
        if (data) {
          if (Buffer.isBuffer(data) || typeof data === 'string') {
            try {
              data = ('' + data).split('\n').map((data) => JSON.parse('' + data));
            } catch (e) { data = {}; }
          }
          Array.isArray(data) || (data = [data]);
        } else {
          data = [{}];
        }
        const worker = selectWorker();
        data.forEach((data: any) => {
          if (worker && !worker.killed) {
            if (!hasListener(worker, 'message', onRender)) { worker.on('message', onRender); }
            data.children = children.map((element) => ReactDOMServer.renderToString(element));
            Object.keys(props).forEach((key) => {
              const val = props[key as keyof Props];
              switch (typeof val) {
                case 'function': break;
                default: data[key] = val;
              }
            });
            worker.send({type: 'render', data, uid: uids[uids.push(nanoid()) - 1]});
          }
        });
      }
    }

    function onRender({type, uid, frame}: any) {
      const idx = uids.indexOf(uid);
      if (idx !== -1) {  //
        uids.splice(idx, 1);
        if (type === 'frame' && frame && peer && peer.connected) {  //
          source.onFrame(frame);
        }
      }
    }

    function selectWorker() {
      const len = workers().length;
      const idx = (len * Math.random()) | 0;
      return workers()[idx];
    }

    function hasListener(ee: NodeJS.EventEmitter, type: string, handler: any) {
      return ee.listeners(type).indexOf(handler) !== -1;
    }
  };

  render = renderRemote;
}

export default render;
