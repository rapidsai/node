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

require('@babel/register')({
  cache: false,
  babelrc: false,
  cwd: __dirname,
  presets: [
    ["@babel/preset-env", { "targets": { "node": "current" } }],
    ['@babel/preset-react', { "useBuiltIns": true }]
  ]
});

const { RapidsJSDOM } = require('@rapidsai/jsdom');

let request = null;

const jsdom = new RapidsJSDOM({
  onAnimationFrameRequested(r) { request = r; }
});

const div = jsdom.window.evalFn(() => {
  return document.body.appendChild(document.createElement('div'));
});

const { Texture2D, Framebuffer, readPixelsToBuffer } = require('@luma.gl/webgl');

const framebuffer = ((_framebuffer) => {
  return (gl) => _framebuffer || (_framebuffer = new Framebuffer(gl, {
    width: 0, height: 0, color: new Texture2D(gl, {
      mipmaps: false,
      parameters: {
        [gl.TEXTURE_MIN_FILTER]: gl.LINEAR,
        [gl.TEXTURE_MAG_FILTER]: gl.LINEAR,
        [gl.TEXTURE_WRAP_S]: gl.CLAMP_TO_EDGE,
        [gl.TEXTURE_WRAP_T]: gl.CLAMP_TO_EDGE,
      }
    })
  }));
})();

const { inspect } = require('util');

process.on('message', ({ type, uid, data }) => {
  if (type === 'exit') {
    return process.exit();
  }
  console.log('worker message:', inspect({ type, uid, data }, true, null, true));
  if (type === 'render') {
    setImmediate(() => process.send({ uid }));
  }
  // const { event, children, ...props } = data;
  // if ('width' in props) { window.outerWidth = props.width; }
  // if ('height' in props) { window.outerHeight = props.height; }
});

console.log('worker started');
