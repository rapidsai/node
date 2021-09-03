#!/usr/bin/env -S node --trace-uncaught

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

const Fastify        = require('fastify');
const webrtcPlugin   = require('./webrtc/plugin');
const {createStream} = require('./webrtc/server');

const dev = process.env.NODE_ENV !== 'production';
// const dev = true;

const fastify = Fastify({logger: {prettyPrint: true, level: 'info'}});

fastify.register(webrtcPlugin, {onConnect: createStream});
fastify.register(require('fastify-nextjs'), {dev});
fastify.after().then(() => { fastify.next('/'); });
// fastify.after(() => {  //
//   fastify.next('/');
// });
// .get('/', (req, reply) => fastify.next('/', req.raw, reply.raw, req.query));

fastify.listen(8080).then(() => console.log('server ready'));
