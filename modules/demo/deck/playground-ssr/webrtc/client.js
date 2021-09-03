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

const Peer = require('simple-peer');
const io   = require('socket.io-client');

function createPeer() {
  const sock = io({transports: ['websocket'], reconnection: false});
  const peer = new Peer({initiator: true, trickle: false, sdpTransform});
  sock.on('signal', (data) => peer.signal(data));
  peer.on('signal', (data) => sock.emit('signal', data));
  return {sock, peer};
  // return negotiate(
  //   rtcId, peer
  //  .on('connect', () => console.log('peer connected'))
  //  .on('close', () => console.log(`peer ${rtcId} close`))
  //  .on('error', (err) => console.error(`peer ${rtcId} error:`, err))
  // );
}

module.exports.createPeer = createPeer;

// const inspect = (obj, indent = '') =>
//   Object.keys(obj)
//     .map((key) => {
//       switch (typeof obj[key]) {
//         case 'object':
//           return `${indent}${key}:\n${obj[key] ? inspect(obj[key], indent + '  ') : 'null'}`;
//         default: return `${indent}${key}: ${obj[key]}`;
//       }
//     })
//     .join('\n');

// function negotiate(rtcId, peer) {
//   return peer.on('signal', async (offer) => {
//     try {
//       // console.log(`peer ${rtcId} offer:`);
//       // console.log(inspect(offer));
//       const response = await fetch('/api/rtc/signal', {
//         method: 'POST',
//         body: JSON.stringify({rtcId, offer}),
//         headers: {'Content-Type': 'application/json'},
//       });
//       if (response.ok) {
//         const answer = await response.json();
//         // console.log(`peer ${rtcId} answer:`);
//         // console.log(inspect(answer));
//         peer.signal(answer);
//       } else {
//         // console.log(`peer ${rtcId} bad answer`);
//         peer.destroy();
//         location = '/';
//       }
//     } catch (e) {
//       console.error(e);
//       peer.destroy();
//       location = '/';
//     }
//   });
// }

function sdpTransform(sdp) {
  // Remove bandwidth restrictions
  // https://github.com/webrtc/samples/blob/89f17a83ed299ef28d45b933419d809b93d41759/src/content/peerconnection/bandwidth/js/main.js#L240
  sdp = sdp.replace(/b=AS:.*\r\n/, '').replace(/b=TIAS:.*\r\n/, '');
  // Force h264 encoding by removing VP8/9 codecs from the sdp
  sdp = onlyH264(sdp);

  return sdp;

  function onlyH264(sdp) {
    // remove non-h264 codecs from the supported codecs list
    const videos = sdp.match(/^m=video.*$/gm);
    if (videos) {
      return videos
               .map((video) => [video,
                                [
                                  ...getCodecIds(sdp, 'VP9'),
                                  ...getCodecIds(sdp, 'VP8'),
                                  ...getCodecIds(sdp, 'HEVC'),
                                  ...getCodecIds(sdp, 'H265')
                                ]])
               .reduce(
                 (sdp, [video, ids]) =>
                   ids
                     .reduce((sdp, id) => [new RegExp(`^a=fmtp:${id}(.*?)$`, 'gm'),
                                           new RegExp(`^a=rtpmap:${id}(.*?)$`, 'gm'),
                                           new RegExp(`^a=rtcp-fb:${id}(.*?)$`, 'gm'),
      ].reduce((sdp, expr) => sdp.replace(expr, ''), sdp),
                             sdp)
                     .replace(video, ids.reduce((video, id) => video.replace(` ${id}`, ''), video)),
                 sdp)
               .replace('\r\n', '\n')
               .split('\n')
               .map((x) => x.trim())
               .filter(Boolean)
               .join('\r\n') +
             '\r\n';
    }

    return sdp;
  }

  function getCodecIds(sdp, codec) {
    return getIdsForMatcher(sdp, new RegExp(`^a=rtpmap:(?<id>\\d+)\\s+${codec}\\/\\d+$`, 'm'))
      .reduce(
        (ids,
         id) => [...ids,
                 id,
                 ...getIdsForMatcher(sdp, new RegExp(`^a=fmtp:(?<id>\\d+)\\s+apt=${id}$`, 'm'))],
        []);
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
