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

import {Deck as BaseDeck} from '@deck.gl/core';
import {createGLContext} from '@luma.gl/gltools';

import {AnimationLoop} from './animation-loop';

export class Deck extends BaseDeck {
  _createAnimationLoop(props: any) {
    const {
      width,
      height,
      gl,
      glOptions,
      debug,
      onError,
      animationProps,
      useDevicePixels,
      createFramebuffer,
      autoResizeDrawingBuffer
    } = props;

    const getFramebufferFromLoop = createFramebuffer && !props._framebuffer;

    const loop = new AnimationLoop({
      _sync: getFramebufferFromLoop,
      width,
      height,
      animationProps,
      useDevicePixels,
      createFramebuffer,
      autoResizeDrawingBuffer,
      autoResizeViewport: false,
      gl,
      onCreateContext: (opts: any) => createGLContext({
        ...glOptions,
        ...opts,
        canvas: this.canvas,
        debug,
        onContextLost: () => this._onContextLost()
      }),
      onInitialize: ({gl}: any)    => this._setGLContext(gl),
      onRender: this._onRenderFrame.bind(this),
      onError
    });

    if (getFramebufferFromLoop) {
      loop.start();
      props._framebuffer      = loop.framebuffer;
      this.props._framebuffer = loop.framebuffer;
    }

    return loop;
  }
}
