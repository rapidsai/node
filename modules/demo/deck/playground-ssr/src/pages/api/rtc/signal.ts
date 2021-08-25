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

import {getPeer} from '../../../render/broker';

export default ((req, response) => {
  return new Promise((resolve) => {
    const {rtcId} = req.body;
    const peer    = getPeer(rtcId);
    response.status(peer ? 200 : 403);
    if (!peer) {
      // debugger;
      console.log(`peer ${rtcId} not found`);
      response.send('Missing or invalid rtcId');
      resolve();
    } else {
      console.log(`peer ${rtcId} offered:`, req.body.offer);
      peer
        .once('signal',
              (data: any) => {
                console.log(`peer ${rtcId} answered:`, data);
                response.json(data);
                resolve();
              })
        .signal(req.body.offer);
    }
  });
}) as import('next').NextApiHandler;
