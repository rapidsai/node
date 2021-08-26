#!/usr/bin/env -S node -r esm

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

const Fastify = require('fastify');

const dev = process.env.NODE_ENV !== 'production';

const fastify = Fastify()
  .register(require('./plugins/webrtc'))
  .register(require('fastify-nextjs'), { dev })
  .after(() => {
    fastify.next('/:rtcId', (next, req, reply) => {
      const { rtcId } = req.params;
      if (!rtcId || !fastify.getPeer(rtcId)) {
        return reply.redirect(302, `/${fastify.newPeer().id}`);
      }
      next.render(req.raw, reply.raw, `/${rtcId}`, req.query);
    });
  });

fastify.listen(8080)
  .then(() => console.log('server ready'));
