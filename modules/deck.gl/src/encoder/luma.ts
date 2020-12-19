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

import { PassThrough } from 'stream';
import { Device } from '@nvidia/cuda';
// import { CUDAEncoderTransform, NvEncoderBufferFormat } from '@nvidia/nvencoder';

interface OnInitializeProps {
  gl: WebGL2RenderingContext;
  canvas: HTMLCanvasElement;
}

export function createAnimationLoopVideoEncoderStream(
  loop: any,
  // @ts-ignore
  {
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    device = new Device(0),
    // format = NvEncoderBufferFormat.ABGR
  },
): Promise<NodeJS.ReadWriteStream> {
  const { readPixelsToBuffer } = require('@luma.gl/webgl');

  let buffer: any = null;
  // let dstData: any = null;
  let frames: NodeJS.ReadWriteStream | null = null;
  // let frames: CUDAEncoderTransform | null = null;

  const {
    onRender: originalOnRender,
    onFinalize: originalOnFinalize,
    onInitialize: originalOnInitialize,
  } = loop.props;

  let onAnimationLoopError: (_?: any) => any = () => {};
  let onAnimationLoopIntialized: (_?: any) => any = () => {};
  const onAnimationLoopIntializedPromise = new Promise((resolve, reject) => {
    onAnimationLoopIntialized = resolve;
    onAnimationLoopError = reject;
  });

  loop.props.onError = onAnimationLoopError;

  loop.props.onInitialize = function onInitialize({ gl, ...props }: OnInitializeProps) {
    // const { width, height } = gl.canvas;

    // frames = new CUDAEncoderTransform({ width, height, format });
    frames = new PassThrough({ objectMode: true, highWaterMark: 1 });

    let result;

    try {
      if (originalOnInitialize) {
        result = originalOnInitialize.call(this, { gl, ...props });
      }
    } catch (e) {
      return onAnimationLoopError(e);
    }

    loop.pause()._addCallbackData(result || {});

    if (!loop.framebuffer) {
      frames.end();
    } else {
      gl.canvas.addEventListener('resize', () => {
        const { width, height } = gl.canvas;
        if (loop.framebuffer) {
          loop.framebuffer.resize({ width, height });
        }
      });
    }

    window.addEventListener('close', () => loop._running && loop.stop());

    onAnimationLoopIntialized();

    return result;
  };

  loop.props.onRender = function onRender(props: { gl: WebGL2RenderingContext }) {
    if (originalOnRender) {
      originalOnRender.call(this, props);
    }
    if (frames && loop._running && loop.framebuffer) {
      const { gl } = props;
      // TODO: flip vertically to account for WebGL's coordinate system (ffmpeg -vf vflip)
      // frames.write({ buffer, format: NvEncoderBufferFormat.ABGR });
      frames.write(<any>{
        width: loop.framebuffer.width,
        height: loop.framebuffer.height,
        data: (buffer = readPixelsToBuffer(loop.framebuffer, {
          target: buffer,
          sourceType: gl.UNSIGNED_BYTE,
        })),
      });
    }
  };

  loop.props.onFinalize = function onFinalize() {
    if (frames) {
      frames.end();
    }
    if (buffer) {
      buffer.delete({ deleteChildren: true });
    }
    frames = buffer = null;
    if (originalOnFinalize) {
      // eslint-disable-next-line prefer-rest-params
      originalOnFinalize.apply(this, arguments);
    }
  };

  // Pause the render loop if running
  loop.pause = function pause() {
    if (this._animationFrameId) {
      cancelAnimationFrame(this._animationFrameId);
    }
    this._nextFramePromise = null;
    this._resolveNextFrame = null;
    this._animationFrameId = null;
    this._running = false;
    return this;
  };

  loop.start();

  return onAnimationLoopIntializedPromise.then(() => frames as NodeJS.ReadWriteStream);
}
