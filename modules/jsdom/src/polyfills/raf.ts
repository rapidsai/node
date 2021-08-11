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

import {glfw} from '@nvidia/glfw';
import * as jsdom from 'jsdom';
import {performance} from 'perf_hooks';

export interface AnimationFrameRequest {
  active: boolean;
  flush: (onAnimationFrameFlushed?: AnimationFrameFlushedCallback) => void;
}

export type AnimationFrameFlushedCallback =
  (startTime: number, flushTime: number, frameTime: number) => any;

export type AnimationFrameRequestedCallback = (request: AnimationFrameRequest) => any;

export function installAnimationFrame(window: jsdom.DOMWindow,
                                      onAnimationFrameRequested: AnimationFrameRequestedCallback) {
  const notMacOs = process.platform !== 'darwin';

  let callbacks0 = new Map<(time: number) => any, any>();
  let callbacks1 = new Map<(time: number) => any, any>();

  let callbacks   = callbacks0;
  const startTime = performance.now();
  const request   = {active: true, flush: flushAnimationFrame};

  return Object.assign(window, {requestAnimationFrame, cancelAnimationFrame});

  function cancelAnimationFrame(cb: (time: number) => any) {
    if (typeof cb === 'function') {
      callbacks.delete(cb);
      request.active = callbacks.size > 0;
    }
  }

  function requestAnimationFrame(cb: (endTime: number) => any = () => {}) {
    if (typeof cb === 'function') {
      callbacks.set(cb, null);
      if (!request.active) {
        request.active = true;
        onAnimationFrameRequested(request);
      }
    }
    return cb;
  }

  function flushAnimationFrame(onAnimationFrameFlushed?: AnimationFrameFlushedCallback) {
    const flushTime = performance.now();
    if (request.active) {
      request.active     = false;
      const initialState = window._clearMask || 0;
      // hack: reset the private `gl._clearMask` field so we know whether
      // to call swapBuffers() after all the listeners have been executed
      window._clearMask = 0;
      if (window.id > 0 && glfw.getCurrentContext() !== window.id) {
        glfw.makeContextCurrent(window.id);
      }
      if (callbacks.size > 0) {
        if (callbacks === callbacks0) {
          callbacks = callbacks1;
          callbacks0.forEach((_, cb) => cb(flushTime - startTime));
          callbacks0 = new Map<(endTime: number) => any, any>();
        } else {
          callbacks = callbacks0;
          callbacks1.forEach((_, cb) => cb(flushTime - startTime));
          callbacks1 = new Map<(endTime: number) => any, any>();
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
    if (typeof onAnimationFrameFlushed === 'function') {
      onAnimationFrameFlushed(startTime, flushTime, performance.now() - flushTime);
    }
  }
}
