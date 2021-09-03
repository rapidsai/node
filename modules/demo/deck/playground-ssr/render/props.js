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

const {CUDA, Uint8Buffer}                          = require('@nvidia/cuda');
const {Buffer: DeckBuffer}                         = require('@rapidsai/deck.gl');
const {Texture2D, Framebuffer, readPixelsToBuffer} = require('@luma.gl/webgl');

const memoizeBuffer = (create) => {
  let resource                         = null;
  return (invalidate = false, ...args) => {
    if (!resource || invalidate) { resource = create(...args); }
    return resource;
  };
};

const rgb_dbuffer = memoizeBuffer((...args) => new DeckBuffer(...args));
const yuv_dbuffer = memoizeBuffer((...args) => new Uint8Buffer(...args));
const yuv_hbuffer = memoizeBuffer((...args) => new Uint8ClampedArray(...args));
const framebuffer = memoizeBuffer((gl, width = 0, height = 0) => new Framebuffer(gl, {
                                    width,
                                    height,
                                    color: new Texture2D(gl, {
                                      mipmaps: false,
                                      parameters: {
                                        [gl.TEXTURE_MIN_FILTER]: gl.LINEAR,
                                        [gl.TEXTURE_MAG_FILTER]: gl.LINEAR,
                                        [gl.TEXTURE_WRAP_S]: gl.CLAMP_TO_EDGE,
                                        [gl.TEXTURE_WRAP_T]: gl.CLAMP_TO_EDGE,
                                      }
                                    })
                                  }))

/**
 *
 * @param {HTMLDivElement} parent
 * @param {Function} onAfterRender
 * @returns
 */
module.exports = (onAfterRender = () => {}) => {
  let _viewState        = null;
  let _interactionState = null;
  let _gl               = document.createElement('canvas').getContext('webgl2');
  let _framebuffer      = framebuffer(false, _gl, window.outerWidth, window.outerHeight);

  return {
    _framebuffer,
    onWebGLInitialized(gl) { console.log('onWebGLInitialized(gl)'); },
    onResize({width, height}) {
      console.log(`onResize(${JSON.stringify({width, height})})`);
      _framebuffer.resize({width, height});
    },
    onViewStateChange({viewState, interactionState, oldViewState}) {
      console.log(`onViewStateChange(${JSON.stringify({viewState, interactionState})})`);
      _viewState        = viewState;
      _interactionState = interactionState;
    },
    onInteractionStateChange(interactionState) {
      console.log(`onInteractionStateChange(${JSON.stringify({interactionState})})`);
      _interactionState = interactionState;
    },
    onAfterRender({gl}) {
      console.log(`onAfterRender({ gl })`);

      let yuv_hbuffer_ = yuv_hbuffer(false);
      let yuv_dbuffer_ = yuv_dbuffer(false);
      let rgb_dbuffer_ = rgb_dbuffer(false, gl);

      const {width, height} = _framebuffer;
      const rgbByteLength   = width * height * 4;
      const yuvByteLength   = width * height * 3 / 2;

      if (rgb_dbuffer_.byteLength < rgbByteLength) {
        rgb_dbuffer_.delete({deleteChildren: true});
        rgb_dbuffer_ = rgb_dbuffer(
          true, gl, {byteLength: rgbByteLength, accessor: {type: gl.UNSIGNED_BYTE, size: 4}});
      }

      if (yuv_dbuffer_.byteLength < yuvByteLength) {
        yuv_dbuffer_ = yuv_dbuffer(true, yuvByteLength);
        yuv_hbuffer_ = yuv_hbuffer(true, yuvByteLength);
      }

      yuv_hbuffer_ = yuv_hbuffer_.subarray(0, yuvByteLength);
      yuv_dbuffer_ = yuv_dbuffer_.subarray(0, yuvByteLength);

      // DtoD copy from framebuffer into our pixelbuffer
      readPixelsToBuffer(
        _framebuffer, {sourceType: gl.UNSIGNED_BYTE, sourceFormat: gl.BGRA, target: rgb_dbuffer_});

      // Map the GL buffer as a CUDA buffer for reading
      rgb_dbuffer_._mapResource();
      ((_rgb_cbuffer) => {
        // flip horizontally to account for WebGL's coordinate system (e.g. ffmpeg -vf vflip)
        CUDA.rgbaMirror(width, height, 0, _rgb_cbuffer);
        // convert colorspace from OpenGL's BGRA to WebRTC's IYUV420
        CUDA.bgraToYCrCb420(yuv_dbuffer_, _rgb_cbuffer, width, height);
      })(rgb_dbuffer_
           .asCUDABuffer()  // Convert the GL buffer to a CUDA Uint8Buffer view
           .subarray(0, rgbByteLength));
      // Unmap the GL buffer's CUDAGraphicsResource
      rgb_dbuffer_._unmapResource();
      // DtoH copy for output
      yuv_dbuffer_.copyInto(yuv_hbuffer_);

      onAfterRender({
        viewState: _viewState,
        interactionState: _interactionState,
        frame: {width, height, data: yuv_hbuffer_}
      });
    },
  };
};
