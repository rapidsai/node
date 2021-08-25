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

const wrtc       = require('wrtc');
const SimplePeer = require('simple-peer');

const peers: Record<string, any> = {};

function delPeer(id: string) {
  console.log(`peer ${id} disconnected`);
  peers[id] && peers[id].destroy();
  delete peers[id];
}

export function getPeer(id: string) { return peers[id]; }

export function newPeer() {
  const id = nanoid();
  console.log(`peer ${id} created`);
  const peer = new SimplePeer({wrtc, sdpTransform})
                 .on('close', delPeer.bind(null, id))
                 .on('error', delPeer.bind(null, id))
                 .on('connect', function onConnect() {  //
                   console.log(`peer ${id} connected`);
                 });
  return {id, peer: peers[id] = peer};
}

function sdpTransform(sdp: string) {
  // Remove bandwidth restrictions
  // https://github.com/webrtc/samples/blob/89f17a83ed299ef28d45b933419d809b93d41759/src/content/peerconnection/bandwidth/js/main.js#L240
  sdp = sdp.replace(/b=AS:.*\r\n/, '').replace(/b=TIAS:.*\r\n/, '');
  return sdp;
}
