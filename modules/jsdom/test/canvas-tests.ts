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

import {RapidsJSDOM} from '@rapidsai/jsdom';
import {evalAsync} from './utils';

test('window.ImageData is from the canvas module', () => {
  const {window} = new RapidsJSDOM();
  expect(window.ImageData).toBe(require('canvas').ImageData);
});

describe('HTMLCanvasElement', () => {
  test(`getContext('webgl2') returns our OpenGL context`, async () => {
    const {window} = new RapidsJSDOM();
    await expect(evalAsync(window, () => {  //
      const gl         = require('@nvidia/webgl');
      const {document} = window;
      const canvas     = document.body.appendChild(document.createElement('canvas'));
      return canvas.getContext('webgl2') instanceof gl.WebGL2RenderingContext;
    })).resolves.toBe(true);
  });

  test(`getContext('webgl2') only creates one OpenGL context`, async () => {
    const {window} = new RapidsJSDOM();
    await expect(evalAsync(window, () => {  //
      const {document} = window;
      const canvas     = document.body.appendChild(document.createElement('canvas'));
      return canvas.getContext('webgl2') === canvas.getContext('webgl2');
    })).resolves.toBe(true);
  });
});
