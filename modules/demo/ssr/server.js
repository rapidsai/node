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

const React = require('react');
const ReactDOM = require('react-dom');
const { App } = require(`${__dirname}/app`);


const { Subject } = require('rxjs');
const { CUDA, devices } = require('@nvidia/cuda');
const { Buffer: DeckBuffer } = require('@rapidsai/deck.gl');
const { Texture2D, Framebuffer, readPixelsToBuffer } = require('@luma.gl/webgl');

module.exports = ({ server, video, input, videoEvents, inputEvents, inputToDOMEvent }) => {

  const { zip, merge, race, interval } = require('rxjs');
  const { filter, finalize, ignoreElements, mergeMap, switchMap, take, takeUntil, tap, } = require('rxjs/operators');

  // When both these streams yield client connections
  return zip(videoEvents.onStreamConnected, inputEvents.onStreamConnected)
    // Map each pair of video/input client connections into a running visualization
    .pipe(mergeMap(([v, i]) => {

      const isThisInput = filter(({ connection }) => connection === i.connection);
      const isThisVideo = filter(({ connection }) => connection === v.connection);
      const inputDisconnected = inputEvents.onStreamDisconnected.pipe(isThisInput);
      const videoDisconnected = videoEvents.onStreamDisconnected.pipe(isThisVideo);
      const clientInputEvents = inputEvents.onClientInputReceived.pipe(isThisInput);

      const { width, height, targetStreamingFps, actualStreamingFps } = video.config.mode;
      const fps = targetStreamingFps || actualStreamingFps;
      const graph = visualize(...capture({
        width, height, layoutParams: { autoCenter: true }
      }));

      const disconnect = race(inputDisconnected, videoDisconnected);

      const inputsLoop = clientInputEvents
        .pipe(mergeMap(({ event }) => Array.isArray(event) ? event : [event]))
        .pipe(tap((event) => graph.dispatchEvent(inputToDOMEvent(window, event))));

      const renderLoop = interval(fps / 1000)
        .pipe(switchMap(() => graph.render().pipe(
          tap(({ width, height, pitch, alpha, pixelbuffer }) => {
            // Map the GL buffer as a CUDA buffer for reading
            pixelbuffer._mapResource();
            // Convert the GL buffer to a CUDA Uint8Buffer view
            const data = pixelbuffer.asCUDABuffer();
            // flip horizontally to account for WebGL's coordinate system (e.g. ffmpeg -vf vflip)
            CUDA.rgbaMirror(width, height, 0, data);
            // Push the GL frame through NvRtcStreamer's NVEncoder pipeline
            v.connection.push({ width, height, pitch, alpha, data: data.buffer.ptr });
            // Unmap the GL buffer's CUDAGraphicsResource
            pixelbuffer._unmapResource();
          })
        )));

      return merge(inputsLoop, renderLoop)
        .pipe(ignoreElements())
        .pipe(takeUntil(disconnect))
        .pipe(finalize(() => graph.close()));
    }))
    .subscribe(); // go!

  function visualize(frames, props = {}) {

    window.outerWidth = props.width;
    window.outerHeight = props.height;

    const { requestAnimationFrame: raf } = window;

    let a = new Map(), b = new Map(), callbacks = a;
    window.requestAnimationFrame = (cb) => { callbacks.set(cb); };
    window.cancelAnimationFrame = (cb) => { callbacks.delete(cb); };

    ReactDOM.render(
      React.createElement(App, {
        ...props,
        ref(e) {
          window._inputEventTarget = ReactDOM.findDOMNode(e);
        }
      }),
      document.body.appendChild(document.createElement('div')));

    return {
      frames: frames.pipe(take(1)),
      render() {
        (callbacks = callbacks === a ? b : a);
        (callbacks === a ? b : a).forEach((_, cb) => raf(cb));
        (callbacks === a) ? (b = new Map()) : (a = new Map());
        return frames;
      },
      dispatchEvent(event) {
        if (event && window && !window.closed) {
          window.dispatchEvent(event);
        }
      },
      close() {
        if (window && !window.closed) {
          window.dispatchEvent({ type: 'close' });
        }
      }
    };
  }

  function capture(props = {}) {

    const frames = new Subject;
    function pitchAlignedSize(lineSize) {
      const { texturePitchAlignment: tpa } = devices[0].getProperties();
      return tpa * (Math.floor(lineSize / tpa) + ((lineSize % tpa) !== 0));
    }

    let framebuffer, pixelbuffer;

    return [
      frames,
      {
        ...props,
        getRenderTarget() { return framebuffer; },
        onWebGLInitialized(gl) {
          pixelbuffer = new DeckBuffer(gl, 0);
          framebuffer = new Framebuffer(gl, {
            width: props.width,
            height: props.height,
            color: new Texture2D(gl, {
              mipmaps: false,
              parameters: {
                [gl.TEXTURE_MIN_FILTER]: gl.LINEAR,
                [gl.TEXTURE_MAG_FILTER]: gl.LINEAR,
                [gl.TEXTURE_WRAP_S]: gl.CLAMP_TO_EDGE,
                [gl.TEXTURE_WRAP_T]: gl.CLAMP_TO_EDGE,
              }
            })
          });
        },
        onResize({ width, height }) {
          if (framebuffer) {
            framebuffer.resize({ width, height });
          }
        },
        onAfterRender({ gl }) {
          const { width, height } = framebuffer;
          const pitch = pitchAlignedSize(width * 4);
          const pitchAlignedByteLength = pitch * height;
          if (pixelbuffer.byteLength !== pitchAlignedByteLength) {
            pixelbuffer.delete({ deleteChildren: true });
            pixelbuffer = new DeckBuffer(gl, {
              byteLength: pitchAlignedByteLength,
              accessor: { type: gl.UNSIGNED_BYTE, size: 4 }
            });
          }
          // DtoD copy from framebuffer into our pixelbuffer
          readPixelsToBuffer(framebuffer, { sourceType: gl.UNSIGNED_BYTE, sourceFormat: gl.BGRA, target: pixelbuffer });
          // console.log({ width, height, pitch, alpha: true, pixelbuffer: pixelbuffer.byteLength });
          frames.next({ width, height, pitch, alpha: true, pixelbuffer });
        }
      }
    ];
  }
};
