// Copyright (c) 2020, NVIDIA CORPORATION.
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

import {performance} from 'perf_hooks';

import {glfw} from '../glfw';

export function installAnimationFrame(window: any) {
  const notMacOs = process.platform !== 'darwin';

  let a = new Map<(time: number) => any, any>();
  let b = new Map<(time: number) => any, any>();

  let callbacks                             = a;
  let animationframe: NodeJS.Immediate|null = null;

  return Object.assign(window, {requestAnimationFrame, cancelAnimationFrame});

  function cancelAnimationFrame(cb: (time: number) => any) {
    if (typeof cb === 'function') {
      callbacks.delete(cb);
      if (animationframe !== null) {
        if (callbacks.size === 0) {
          const af       = animationframe;
          animationframe = null;
          clearImmediate(af);
        }
      }
    }
  }

  function requestAnimationFrame(cb: (time: number) => any = () => {}) {
    if (animationframe === null) { animationframe = setImmediate(flushAnimationFrame); }
    callbacks.set(cb, null);
    return cb;
  }

  function flushAnimationFrame() {
    animationframe     = null;
    const initialState = window._clearMask || 0;
    // hack: reset the private `gl._clearMask` field so we know whether
    // to call swapBuffers() after all the listeners have been executed
    window._clearMask = 0;
    (window.id > 0) && glfw.makeContextCurrent(window.id);
    if (callbacks.size > 0) {
      const t_ = performance.now();
      if (callbacks === a) {
        callbacks = b;
        a.forEach((_, cb) => cb(t_));
        a = new Map<(time: number) => any, any>();
      } else {
        callbacks = a;
        b.forEach((_, cb) => cb(t_));
        b = new Map<(time: number) => any, any>();
      }
    }
    const resultState = window._clearMask || 0;
    window._clearMask = 0;
    // Fix for MacOS: only swap buffers if gl.clear() was called
    if (notMacOs || (initialState || resultState)) {
      (window.id > 0) && glfw.swapBuffers(window.id);
    }
    glfw.pollEvents();
  }
}
