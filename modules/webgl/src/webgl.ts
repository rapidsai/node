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

/* eslint-disable prefer-const */

import gl from './addon';

// default WebGLContextAttributes:
//     alpha: true,
//     antialias: true,
//     depth: true,
//     failIfMajorPerformanceCaveat: false,
//     powerPreference: 'default',
//     premultipliedAlpha: true,
//     preserveDrawingBuffer: false,
//     stencil: false,
//     desynchronized: false

interface OpenGLESRenderingContext extends WebGL2RenderingContext {
  // eslint-disable-next-line @typescript-eslint/no-misused-new
  new(attrs?: WebGLContextAttributes): OpenGLESRenderingContext;
  webgl1: boolean;
  webgl2: boolean;
  opengl: boolean;
  _version: number;
  _clearMask: number;
}

// eslint-disable-next-line @typescript-eslint/no-redeclare
const OpenGLESRenderingContext: OpenGLESRenderingContext = gl.WebGL2RenderingContext;

OpenGLESRenderingContext.prototype.webgl1   = false;
OpenGLESRenderingContext.prototype.webgl2   = true;
OpenGLESRenderingContext.prototype.opengl   = true;
OpenGLESRenderingContext.prototype._version = 2;

export const WebGLActiveInfo            = gl.WebGLActiveInfo;
export const WebGLShaderPrecisionFormat = gl.WebGLShaderPrecisionFormat;
export const WebGLBuffer                = gl.WebGLBuffer;
export const WebGLContextEvent          = gl.WebGLContextEvent;
export const WebGLFramebuffer           = gl.WebGLFramebuffer;
export const WebGLProgram               = gl.WebGLProgram;
export const WebGLQuery                 = gl.WebGLQuery;
export const WebGLRenderbuffer          = gl.WebGLRenderbuffer;
export const WebGLSampler               = gl.WebGLSampler;
export const WebGLShader                = gl.WebGLShader;
export const WebGLSync                  = gl.WebGLSync;
export const WebGLTexture               = gl.WebGLTexture;
export const WebGLTransformFeedback     = gl.WebGLTransformFeedback;
export const WebGLUniformLocation       = gl.WebGLUniformLocation;
export const WebGLVertexArrayObject     = gl.WebGLVertexArrayObject;
export {OpenGLESRenderingContext as WebGLRenderingContext};
export {OpenGLESRenderingContext as WebGL2RenderingContext};

const gl_bufferData                           = OpenGLESRenderingContext.prototype.bufferData;
OpenGLESRenderingContext.prototype.bufferData = bufferData;
function bufferData(
  this: WebGL2RenderingContext, target: GLenum, size: GLsizeiptr, usage: GLenum): void;
function bufferData(
  this: WebGL2RenderingContext, target: GLenum, srcData: BufferSource|null, usage: GLenum): void;
function bufferData(this: WebGL2RenderingContext,
                    target: GLenum,
                    srcData: ArrayBufferView,
                    usage: GLenum,
                    srcOffset: GLuint,
                    srcByteLength?: GLuint): void;
function bufferData(this: WebGL2RenderingContext, ...args: [GLenum, GLsizeiptr | BufferSource | null, GLenum, GLuint?, GLuint?]): void {
  let [target, src, usage, srcOffset, srcByteLength] = args;
  if (args.length > 3 && src !== null && typeof src !== 'number' && typeof srcOffset === 'number') {
    let BPM, arr = ArrayBuffer.isView(src) ? src : new Uint8Array(src);
    [, , , , srcByteLength = arr.byteLength, BPM = (<Uint8Array>arr).BYTES_PER_ELEMENT] = args;
    src = new Uint8Array(arr.buffer, arr.byteOffset, srcByteLength).subarray(srcOffset * BPM);
  }
  return gl_bufferData.call(this, target, src, usage);
}

const gl_bufferSubData                           = OpenGLESRenderingContext.prototype.bufferSubData;
OpenGLESRenderingContext.prototype.bufferSubData = bufferSubData;
function bufferSubData(
  this: WebGL2RenderingContext, target: GLenum, offset: GLintptr, data: BufferSource): void;
function bufferSubData(
  this: WebGL2RenderingContext, target: GLenum, dstByteOffset: GLintptr, srcData: BufferSource):
  void;
function bufferSubData(this: WebGL2RenderingContext,
                       target: GLenum,
                       dstByteOffset: GLintptr,
                       srcData: ArrayBufferView,
                       srcOffset: GLuint,
                       srcByteLength?: GLuint): void;
function bufferSubData(this: WebGL2RenderingContext, ...args: [GLenum, GLintptr, BufferSource | ArrayBufferView, GLuint?, GLuint?]): void {
  let [target, dstByteOffset, src, srcOffset, srcByteLength] = args;
  if (args.length > 3 && src !== null && typeof src !== 'number' && typeof srcOffset === 'number') {
    let BPM, arr = ArrayBuffer.isView(src) ? src : new Uint8Array(src);
    [, , , , srcByteLength = arr.byteLength, BPM = (<Uint8Array>arr).BYTES_PER_ELEMENT] = args;
    src = new Uint8Array(arr.buffer, arr.byteOffset, srcByteLength).subarray(srcOffset * BPM);
  }
  return gl_bufferSubData.call(this, target, dstByteOffset, src);
}

const gl_getBufferSubData = OpenGLESRenderingContext.prototype.getBufferSubData;
OpenGLESRenderingContext.prototype.getBufferSubData = getBufferSubData;
function getBufferSubData(this: WebGL2RenderingContext,
                          target: GLenum,
                          srcByteOffset: GLintptr,
                          dst: ArrayBufferView,
                          dstOffset: GLuint = 0,
                          length?: GLuint): void {
  // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
  const arr  = ArrayBuffer.isView(dst) ? <Uint8Array>dst : toArrayBufferViewSlice(dst, dstOffset)!;
  const size = typeof length === 'undefined' ? arr.byteLength : length * arr.BYTES_PER_ELEMENT;
  return gl_getBufferSubData.call(this, target, srcByteOffset, size, arr);
}

// @ts-ignore
// const gl_getExtension = OpenGLESRenderingContext.prototype.getExtension;
OpenGLESRenderingContext.prototype.getExtension = getExtension;
function getExtension(this: WebGL2RenderingContext, name: string) { return extensionsMap [name]; }

const gl_getShaderInfoLog = OpenGLESRenderingContext.prototype.getShaderInfoLog;
OpenGLESRenderingContext.prototype.getShaderInfoLog = getShaderInfoLog;
function getShaderInfoLog(this: WebGL2RenderingContext, shader: WebGLShader): string {
  const lines: string [] = (gl_getShaderInfoLog.call(this, shader) || '').split(/(\n|\r)/g);
  // Reformat the error lines to look like webgl errors (todo: is this nvidia-specific?)
  return lines
    .map((line) => {
      let errIndex, numIndex, errMatch, numMatch, type, num;
      ({index: errIndex, [0]: errMatch, [1]: type} =
         // eslint-disable-next-line @typescript-eslint/prefer-regexp-exec
       (lines [0] || '').match(/\s?(warning|error) (\w+|\d+):/) || {...['', '']});
      ({index: numIndex, [0]: numMatch, [1]: num} =
         // eslint-disable-next-line @typescript-eslint/prefer-regexp-exec
       (line.match(/0\((\d+)\) :/) || {...['', '']}));
      if (errIndex !== undefined && numIndex !== undefined && errMatch && type && numMatch && num) {
        return `${type.toUpperCase()}:${line.slice(errIndex + errMatch.length)}:${num}${
          line.slice(numIndex + numMatch.length)}`;
      }
      return line;
    })
    .join('\n');
}

const gl_texImage2D                           = OpenGLESRenderingContext.prototype.texImage2D;
OpenGLESRenderingContext.prototype.texImage2D = texImage2D;
function texImage2D(this: WebGL2RenderingContext,
                    target: GLenum,
                    level: GLint,
                    internalformat: GLint,
                    format: GLenum,
                    type: GLenum,
                    source: TexImageSource): void;
function texImage2D(this: WebGL2RenderingContext,
                    target: GLenum,
                    level: GLint,
                    internalformat: GLint,
                    width: GLsizei,
                    height: GLsizei,
                    border: GLint,
                    format: GLenum,
                    type: GLenum,
                    pboOffset: GLintptr): void;
function texImage2D(this: WebGL2RenderingContext,
                    target: GLenum,
                    level: GLint,
                    internalformat: GLint,
                    width: GLsizei,
                    height: GLsizei,
                    border: GLint,
                    format: GLenum,
                    type: GLenum,
                    source: TexImageSource): void;
function texImage2D(this: WebGL2RenderingContext,
                    target: GLenum,
                    level: GLint,
                    internalformat: GLint,
                    width: GLsizei,
                    height: GLsizei,
                    border: GLint,
                    format: GLenum,
                    type: GLenum,
                    srcData?: ArrayBufferView|null,
                    srcOffset?: GLuint): void;
function texImage2D(this: WebGL2RenderingContext, ...args: [GLenum, GLint, GLint, GLsizei | GLenum, GLsizei | GLenum, GLint | TexImageSource, GLenum?, GLenum?, (GLintptr | TexImageSource | ArrayBufferView | null)?, GLsizei?]): void {
  let [target, level, internalformat, width, height, border, format, type, src, offset = 0] = args;
  switch (args.length) {
    case 6: {
      [target, level, internalformat, format, type, src] =
        (args as [GLenum, GLint, GLint, GLenum, GLenum, TexImageSource]);
      ({width, height, border = 0} = <any>src);
      src = pixelsFromImage(src, width, height);
      break;
    }
    case 8:
    case 9:
    case 10: {
      if (typeof args [8] === 'number') {
        [target, level, internalformat, width, height, border, format, type, src] =
          (args as [GLenum, GLint, GLint, GLsizei, GLsizei, GLint, GLenum, GLenum, GLintptr]);
        break;
      }
      if (args [8] === null || args [8] === undefined || ArrayBuffer.isView(args [8]) ||
          (args [8] instanceof ArrayBuffer)) {
        [target, level, internalformat, width, height, border, format, type, src] =
          (args as
             [GLenum, GLint, GLint, GLsizei, GLsizei, GLint, GLenum, GLenum, ArrayBufferView]);
        src = toArrayBufferViewSlice(src, offset);
        break;
      }
      if (args [8] && typeof args [8] === 'object') {
        [target, level, internalformat, width, height, border, format, type, src] =
          (args as [GLenum, GLint, GLint, GLsizei, GLsizei, GLint, GLenum, GLenum, TexImageSource]);
        [width = src.width, height = src.height] = [width, height];
        src                                      = pixelsFromImage(src, width, height);
        break;
      }
      throw new TypeError('WebGLRenderingContext texImage2D() invalid texture source');
    }
    default: throw new TypeError('WebGLRenderingContext texImage2D() takes 6, 9, or 10 arguments');
  }
  return gl_texImage2D.call(
    this, target, level, internalformat, width, height, border, format, type, src);
}

const gl_texSubImage2D                           = OpenGLESRenderingContext.prototype.texSubImage2D;
OpenGLESRenderingContext.prototype.texSubImage2D = texSubImage2D;
function texSubImage2D(this: WebGL2RenderingContext,
                       target: GLenum,
                       level: GLint,
                       xoffset: GLint,
                       yoffset: GLint,
                       width: GLsizei,
                       height: GLsizei,
                       format: GLenum,
                       type: GLenum,
                       pixels: ArrayBufferView|null): void;
function texSubImage2D(this: WebGL2RenderingContext,
                       target: GLenum,
                       level: GLint,
                       xoffset: GLint,
                       yoffset: GLint,
                       format: GLenum,
                       type: GLenum,
                       source: TexImageSource): void;
function texSubImage2D(this: WebGL2RenderingContext,
                       target: GLenum,
                       level: GLint,
                       xoffset: GLint,
                       yoffset: GLint,
                       width: GLsizei,
                       height: GLsizei,
                       format: GLenum,
                       type: GLenum,
                       pboOffset: GLintptr): void;
function texSubImage2D(this: WebGL2RenderingContext,
                       target: GLenum,
                       level: GLint,
                       xoffset: GLint,
                       yoffset: GLint,
                       width: GLsizei,
                       height: GLsizei,
                       format: GLenum,
                       type: GLenum,
                       source: TexImageSource): void;
function texSubImage2D(this: WebGL2RenderingContext,
                       target: GLenum,
                       level: GLint,
                       xoffset: GLint,
                       yoffset: GLint,
                       width: GLsizei,
                       height: GLsizei,
                       format: GLenum,
                       type: GLenum,
                       srcData?: ArrayBufferView|null,
                       srcOffset?: GLuint): void;
function texSubImage2D(this: WebGL2RenderingContext, ...args: [GLenum, GLint, GLint, GLint, GLenum | GLsizei, GLenum | GLsizei, GLenum | TexImageSource, GLenum?, (GLintptr | TexImageSource | ArrayBufferView | null)?, GLuint?]): void {
  let [target, level, x, y, width, height, format, type, src, offset = 0] = args;
  switch (args.length) {
    case 7: {
      [target, level, x, y, format, type, src] =
        (args as [GLenum, GLint, GLint, GLint, GLenum, GLenum, TexImageSource]);
      src = pixelsFromImage(src, width = src.width, height = src.height);
      break;
    }
    case 8:
    case 9:
    case 10: {
      if (typeof args [8] === 'number') {
        [target, level, x, y, width, height, format, type, src] =
          (args as [GLenum, GLint, GLint, GLint, GLsizei, GLsizei, GLenum, GLenum, GLintptr]);
        break;
      }
      if (args [8] === null || args [8] === undefined || ArrayBuffer.isView(args [8]) ||
          (args [8] instanceof ArrayBuffer)) {
        [target, level, x, y, width, height, format, type, src] =
          (args as [GLenum,
                    GLint,
                    GLint,
                    GLint,
                    GLsizei,
                    GLsizei,
                    GLenum,
                    GLenum,
                    ArrayBufferView | null]);
        src = toArrayBufferViewSlice(src, offset);
        break;
      }
      if (args [8] && typeof args [8] === 'object') {
        [target, level, x, y, width, height, format, type, src] =
          (args as [GLenum, GLint, GLint, GLint, GLsizei, GLsizei, GLenum, GLenum, TexImageSource]);
        [width = src.width, height = src.height] = [width, height];
        src                                      = pixelsFromImage(src, width, height);
        break;
      }
      throw new TypeError('WebGLRenderingContext texSubImage2D() invalid texture source');
    }
    default:
      throw new TypeError('WebGLRenderingContext texSubImage2D() takes 7, 9, or 10 arguments');
  }
  return gl_texSubImage2D.call(this, target, level, x, y, width, height, format, type, src);
}

const gl_readPixels                           = OpenGLESRenderingContext.prototype.readPixels;
OpenGLESRenderingContext.prototype.readPixels = readPixels;
function readPixels(this: WebGL2RenderingContext,
                    x: GLint,
                    y: GLint,
                    width: GLsizei,
                    height: GLsizei,
                    format: GLenum,
                    type: GLenum,
                    dstData: ArrayBufferView|null): void;
function readPixels(this: WebGL2RenderingContext,
                    x: GLint,
                    y: GLint,
                    width: GLsizei,
                    height: GLsizei,
                    format: GLenum,
                    type: GLenum,
                    offset: GLintptr): void;
function readPixels(this: WebGL2RenderingContext,
                    x: GLint,
                    y: GLint,
                    width: GLsizei,
                    height: GLsizei,
                    format: GLenum,
                    type: GLenum,
                    dstData: ArrayBufferView,
                    dstOffset: GLuint): void;
function readPixels(this: WebGL2RenderingContext, ...args: [GLint, GLint, GLsizei, GLsizei, GLenum, GLenum, (ArrayBufferView | ArrayBuffer | GLintptr | null)?, GLuint?]): void {
  let [x, y, width, height, format, type, dst, offset = 0] = args;
  switch (args.length) {
    case 6:
    case 7: {
      if (typeof args [6] === 'number') {
        [x, y, width, height, format, type, dst] =
          (args as [GLint, GLint, GLsizei, GLsizei, GLenum, GLenum, GLintptr]);
        break;
      }
      if (args [6] === null || args [6] === undefined || ArrayBuffer.isView(args [6]) ||
          (args [6] instanceof ArrayBuffer)) {
                [x, y, width, height, format, type, dst] = (args as [GLint, GLint, GLsizei, GLsizei, GLenum, GLenum, (ArrayBufferView | null)?]);
                dst                                      = toArrayBufferViewSlice(dst, offset);
                break;
      }
      throw new TypeError('WebGLRenderingContext readPixels() invalid readPixels target');
    }
    case 8: {
            [x, y, width, height, format, type, dst, offset = 0] = (args as [GLint, GLint, GLsizei, GLsizei, GLenum, GLenum, ArrayBufferView?, GLuint?]);
            dst = toArrayBufferViewSlice(dst, offset);
            break;
    }
    default: throw new TypeError('WebGLRenderingContext readPixels() takes 6, 7, or 8 arguments');
  }
  return gl_readPixels.call(this, x, y, width, height, format, type, dst);
}

const gl_uniformMatrix2fv = OpenGLESRenderingContext.prototype.uniformMatrix2fv;
OpenGLESRenderingContext.prototype.uniformMatrix2fv = uniformMatrix2fv;
function uniformMatrix2fv(this: WebGL2RenderingContext,
                          location: WebGLUniformLocation|null,
                          transpose: GLboolean,
                          data: Float32List,
                          offset: GLuint = 0,
                          length: GLuint = data.length): void {
  if (ArrayBuffer.isView(data)) { data = data.subarray(offset, offset + length); }
  return location ? gl_uniformMatrix2fv.call(this, location, transpose, data) : undefined;
}

const gl_uniformMatrix3fv = OpenGLESRenderingContext.prototype.uniformMatrix3fv;
OpenGLESRenderingContext.prototype.uniformMatrix3fv = uniformMatrix3fv;
function uniformMatrix3fv(this: WebGL2RenderingContext,
                          location: WebGLUniformLocation|null,
                          transpose: GLboolean,
                          data: Float32List,
                          offset: GLuint = 0,
                          length: GLuint = data.length): void {
  if (ArrayBuffer.isView(data)) { data = data.subarray(offset, offset + length); }
  return location ? gl_uniformMatrix3fv.call(this, location, transpose, data) : undefined;
}

const gl_uniformMatrix4fv = OpenGLESRenderingContext.prototype.uniformMatrix4fv;
OpenGLESRenderingContext.prototype.uniformMatrix4fv = uniformMatrix4fv;
function uniformMatrix4fv(this: WebGL2RenderingContext,
                          location: WebGLUniformLocation|null,
                          transpose: GLboolean,
                          data: Float32List,
                          offset: GLuint = 0,
                          length: GLuint = data.length): void {
  if (ArrayBuffer.isView(data)) { data = data.subarray(offset, offset + length); }
  return location ? gl_uniformMatrix4fv.call(this, location, transpose, data) : undefined;
}

const gl_uniformMatrix2x3fv = OpenGLESRenderingContext.prototype.uniformMatrix2x3fv;
OpenGLESRenderingContext.prototype.uniformMatrix2x3fv = uniformMatrix2x3fv;
function uniformMatrix2x3fv(this: WebGL2RenderingContext,
                            location: WebGLUniformLocation|null,
                            transpose: GLboolean,
                            data: Float32List,
                            offset: GLuint = 0,
                            length: GLuint = data.length): void {
  if (ArrayBuffer.isView(data)) { data = data.subarray(offset, offset + length); }
  return location ? gl_uniformMatrix2x3fv.call(this, location, transpose, data) : undefined;
}

const gl_uniformMatrix2x4fv = OpenGLESRenderingContext.prototype.uniformMatrix2x4fv;
OpenGLESRenderingContext.prototype.uniformMatrix2x4fv = uniformMatrix2x4fv;
function uniformMatrix2x4fv(this: WebGL2RenderingContext,
                            location: WebGLUniformLocation|null,
                            transpose: GLboolean,
                            data: Float32List,
                            offset: GLuint = 0,
                            length: GLuint = data.length): void {
  if (ArrayBuffer.isView(data)) { data = data.subarray(offset, offset + length); }
  return location ? gl_uniformMatrix2x4fv.call(this, location, transpose, data) : undefined;
}

const gl_uniformMatrix3x2fv = OpenGLESRenderingContext.prototype.uniformMatrix3x2fv;
OpenGLESRenderingContext.prototype.uniformMatrix3x2fv = uniformMatrix3x2fv;
function uniformMatrix3x2fv(this: WebGL2RenderingContext,
                            location: WebGLUniformLocation|null,
                            transpose: GLboolean,
                            data: Float32List,
                            offset: GLuint = 0,
                            length: GLuint = data.length): void {
  if (ArrayBuffer.isView(data)) { data = data.subarray(offset, offset + length); }
  return location ? gl_uniformMatrix3x2fv.call(this, location, transpose, data) : undefined;
}

const gl_uniformMatrix3x4fv = OpenGLESRenderingContext.prototype.uniformMatrix3x4fv;
OpenGLESRenderingContext.prototype.uniformMatrix3x4fv = uniformMatrix3x4fv;
function uniformMatrix3x4fv(this: WebGL2RenderingContext,
                            location: WebGLUniformLocation|null,
                            transpose: GLboolean,
                            data: Float32List,
                            offset: GLuint = 0,
                            length: GLuint = data.length): void {
  if (ArrayBuffer.isView(data)) { data = data.subarray(offset, offset + length); }
  return location ? gl_uniformMatrix3x4fv.call(this, location, transpose, data) : undefined;
}

const gl_uniformMatrix4x2fv = OpenGLESRenderingContext.prototype.uniformMatrix4x2fv;
OpenGLESRenderingContext.prototype.uniformMatrix4x2fv = uniformMatrix4x2fv;
function uniformMatrix4x2fv(this: WebGL2RenderingContext,
                            location: WebGLUniformLocation|null,
                            transpose: GLboolean,
                            data: Float32List,
                            offset: GLuint = 0,
                            length: GLuint = data.length): void {
  if (ArrayBuffer.isView(data)) { data = data.subarray(offset, offset + length); }
  return location ? gl_uniformMatrix4x2fv.call(this, location, transpose, data) : undefined;
}

const gl_uniformMatrix4x3fv = OpenGLESRenderingContext.prototype.uniformMatrix4x3fv;
OpenGLESRenderingContext.prototype.uniformMatrix4x3fv = uniformMatrix4x3fv;
function uniformMatrix4x3fv(this: WebGL2RenderingContext,
                            location: WebGLUniformLocation|null,
                            transpose: GLboolean,
                            data: Float32List,
                            offset: GLuint = 0,
                            length: GLuint = data.length): void {
  if (ArrayBuffer.isView(data)) { data = data.subarray(offset, offset + length); }
  return location ? gl_uniformMatrix4x3fv.call(this, location, transpose, data) : undefined;
}

if (Boolean(process.env.NVIDIA_NODE_WEBGL_TRACE_CALLS) === true) {
  wrapAndLogGLMethods(OpenGLESRenderingContext.prototype);
}

//@ts-ignore
function wrapAndLogGLMethods(proto: any) {
  /* eslint-disable @typescript-eslint/restrict-template-expressions */
  const listToString = (x: any) => {
    if (x.length < 10) { return `(length=${x.length}, values=[${x}])`; }
    return `(length=${x.length}, values=[${x.slice(0, 3)}, ... ${
      x.slice((x.length >> 1) - 2, (x.length >> 1) + 1)}, ... ${x.slice(x.length - 3, x.length)}])`
  };
  const toString = (x: any) => {
    switch (typeof x) {
      case 'number': return `${x}`;
      case 'boolean': return `${x}`;
      case 'function': return `${x}`;
      case 'string': return `\`${x}\``;
      case 'undefined': return 'undefined';
      case 'symbol': return `Symbol(${x.description})`;
      default:
        if (x === null) return 'null';
        if (Array.isArray(x)) return listToString(x);
        if (ArrayBuffer.isView(x)) return `${x.constructor.name}${listToString(x)}`;
        if (nodeCustomInspectSym in x) return x [nodeCustomInspectSym]();
        if (`${x}` === '[object Object]') return JSON.stringify(x);
        return `${x}`;
    }
  };
  const logArguments = (name: string, fn: any) => function(this: any, ...args: any []) {
    const str = `gl.${name}(${args.map(toString).join(', ')})`;
    process.stderr.write(str);
    try {
      const ret = fn.apply(this, args);
      process.stderr.write((ret ? `: ${toString(ret)}` : '') + '\n');
      return ret;
    } catch (err) {
      process.stderr.write((err ? `: ${toString(err)}` : '') + '\n');
      throw err;
    }
  };
  Object.keys(proto)
    .filter((key: any) => typeof proto [key] === 'function')
    .forEach((key) => proto [key] = logArguments(key, proto [key]));
}

function toArrayBufferViewSlice(source?: ArrayBuffer|ArrayBufferView|null, offset = 0) {
  if (source === null || source === undefined) { return null; }
  if (source instanceof ArrayBuffer) {
    return new Uint8Array(source).subarray(offset);
  } else if (source instanceof DataView) {
    return new Uint8Array(source.buffer).subarray(offset);
  } else if (ArrayBuffer.isView(source)) {
    return (<Uint8Array>source).subarray(offset * (<Uint8Array>source).BYTES_PER_ELEMENT);
  }
  throw new TypeError('OpenGLESRenderingContext invalid pixel source');
}

function pixelsFromImage(source: TexImageSource, width: number, height: number) {
  if ((typeof HTMLImageElement !== 'undefined') && (source instanceof HTMLImageElement) ||
      (typeof HTMLVideoElement !== 'undefined') && (source instanceof HTMLVideoElement) &&
        source.ownerDocument) {
    const canvas = source.ownerDocument.createElement('canvas');
    // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
    const context = Object.assign(canvas, {width, height}).getContext('2d')!;
    context.drawImage(source, 0, 0);
    return context.getImageData(0, 0, width, height).data;
  } else if ((typeof OffscreenCanvas !== 'undefined') && (source instanceof OffscreenCanvas) ||
             (typeof HTMLCanvasElement !== 'undefined') && (source instanceof HTMLCanvasElement)) {
    // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
    return source.getContext('2d')!.getImageData(0, 0, width, height).data;
  } else if ('data' in source) {
    return source.data;
  }
  throw new TypeError('OpenGLESRenderingContext invalid pixel source');
}

const nodeCustomInspectSym = Symbol.for('nodejs.util.inspect.custom');

const extensionsMap: any = {
  ANGLE_instanced_arrays: {
    VERTEX_ATTRIB_ARRAY_DIVISOR_ANGLE: gl.VERTEX_ATTRIB_ARRAY_DIVISOR,
    drawArraysInstancedANGLE: gl.drawArraysInstanced,
    drawElementsInstancedANGLE: gl.drawElementsInstanced,
    vertexAttribDivisorANGLE: gl.vertexAttribDivisor,
  },
  EXT_blend_minmax: {
    MIN_EXT: gl.MIN,
    MAX_EXT: gl.MAX,
  },
  EXT_color_buffer_float: {},
  EXT_color_buffer_half_float: {
    RGBA16F_EXT: gl.RGBA16F,
    RGB16F_EXT: gl.RGB16F,
    FRAMEBUFFER_ATTACHMENT_COMPONENT_TYPE_EXT: gl.FRAMEBUFFER_ATTACHMENT_COMPONENT_TYPE,
    UNSIGNED_NORMALIZED_EXT: gl.UNSIGNED_NORMALIZED,
  },
  EXT_disjoint_timer_query: {
    QUERY_COUNTER_BITS_EXT: gl.QUERY_COUNTER_BITS,
    CURRENT_QUERY_EXT: gl.CURRENT_QUERY,
    QUERY_RESULT_EXT: gl.QUERY_RESULT,
    QUERY_RESULT_AVAILABLE_EXT: gl.QUERY_RESULT_AVAILABLE,
    TIME_ELAPSED_EXT: gl.TIME_ELAPSED,
    TIMESTAMP_EXT: gl.TIMESTAMP,
    GPU_DISJOINT_EXT: gl.GPU_DISJOINT,
    createQueryEXT: gl.createQuery,
    deleteQueryEXT: gl.deleteQuery,
    isQueryEXT: gl.isQuery,
    beginQueryEXT: gl.beginQuery,
    endQueryEXT: gl.endQuery,
    queryCounterEXT: gl.queryCounter,
    getQueryEXT: gl.getQuery,
    getQueryObjectEXT: gl.getQueryObject,
  },
  EXT_frag_depth: {},
  EXT_sRGB: {
    SRGB_EXT: gl.SRGB,
    SRGB_ALPHA_EXT: gl.SRGB_ALPHA,
    SRGB8_ALPHA8_EXT: gl.SRGB8_ALPHA8,
    FRAMEBUFFER_ATTACHMENT_COLOR_ENCODING_EXT: gl.FRAMEBUFFER_ATTACHMENT_COLOR_ENCODING,
  },
  EXT_shader_texture_lod: {},
  EXT_texture_filter_anisotropic: {
    MAX_TEXTURE_MAX_ANISOTROPY_EXT: gl.MAX_TEXTURE_MAX_ANISOTROPY,
    TEXTURE_MAX_ANISOTROPY_EXT: gl.TEXTURE_MAX_ANISOTROPY,
  },
  OES_element_index_uint: {},
  OES_standard_derivatives: {
    FRAGMENT_SHADER_DERIVATIVE_HINT_OES: gl.FRAGMENT_SHADER_DERIVATIVE_HINT,
  },
  OES_texture_float: {},
  OES_texture_float_linear: {},
  OES_texture_half_float: {
    HALF_FLOAT_OES: gl.HALF_FLOAT,
  },
  OES_texture_half_float_linear: {},
  OES_vertex_array_object: {
    VERTEX_ARRAY_BINDING_OES: gl.VERTEX_ARRAY_BINDING,
    createVertexArrayOES: gl.createVertexArray,
    deleteVertexArrayOES: gl.deleteVertexArray,
    isVertexArrayOES: gl.isVertexArray,
    bindVertexArrayOES: gl.bindVertexArray,
  },
  WEBGL_color_buffer_float: {
    RGBA32F_EXT: gl.RGBA32F,
    RGB32F_EXT: gl.RGB32F,
    FRAMEBUFFER_ATTACHMENT_COMPONENT_TYPE_EXT: gl.FRAMEBUFFER_ATTACHMENT_COMPONENT_TYPE,
    UNSIGNED_NORMALIZED_EXT: gl.UNSIGNED_NORMALIZED,
  },
  WEBGL_compressed_texture_astc: {
    getSupportedProfiles() {},
    COMPRESSED_RGBA_ASTC_4x4_KHR: gl.COMPRESSED_RGBA_ASTC_4x4,
    COMPRESSED_RGBA_ASTC_5x4_KHR: gl.COMPRESSED_RGBA_ASTC_5x4,
    COMPRESSED_RGBA_ASTC_5x5_KHR: gl.COMPRESSED_RGBA_ASTC_5x5,
    COMPRESSED_RGBA_ASTC_6x5_KHR: gl.COMPRESSED_RGBA_ASTC_6x5,
    COMPRESSED_RGBA_ASTC_6x6_KHR: gl.COMPRESSED_RGBA_ASTC_6x6,
    COMPRESSED_RGBA_ASTC_8x5_KHR: gl.COMPRESSED_RGBA_ASTC_8x5,
    COMPRESSED_RGBA_ASTC_8x6_KHR: gl.COMPRESSED_RGBA_ASTC_8x6,
    COMPRESSED_RGBA_ASTC_8x8_KHR: gl.COMPRESSED_RGBA_ASTC_8x8,
    COMPRESSED_RGBA_ASTC_10x5_KHR: gl.COMPRESSED_RGBA_ASTC_10x5,
    COMPRESSED_RGBA_ASTC_10x6_KHR: gl.COMPRESSED_RGBA_ASTC_10x6,
    COMPRESSED_RGBA_ASTC_10x8_KHR: gl.COMPRESSED_RGBA_ASTC_10x8,
    COMPRESSED_RGBA_ASTC_10x10_KHR: gl.COMPRESSED_RGBA_ASTC_10x10,
    COMPRESSED_RGBA_ASTC_12x10_KHR: gl.COMPRESSED_RGBA_ASTC_12x10,
    COMPRESSED_RGBA_ASTC_12x12_KHR: gl.COMPRESSED_RGBA_ASTC_12x12,
    COMPRESSED_SRGB8_ALPHA8_ASTC_4x4_KHR: gl.COMPRESSED_SRGB8_ALPHA8_ASTC_4x4,
    COMPRESSED_SRGB8_ALPHA8_ASTC_5x4_KHR: gl.COMPRESSED_SRGB8_ALPHA8_ASTC_5x4,
    COMPRESSED_SRGB8_ALPHA8_ASTC_5x5_KHR: gl.COMPRESSED_SRGB8_ALPHA8_ASTC_5x5,
    COMPRESSED_SRGB8_ALPHA8_ASTC_6x5_KHR: gl.COMPRESSED_SRGB8_ALPHA8_ASTC_6x5,
    COMPRESSED_SRGB8_ALPHA8_ASTC_6x6_KHR: gl.COMPRESSED_SRGB8_ALPHA8_ASTC_6x6,
    COMPRESSED_SRGB8_ALPHA8_ASTC_8x5_KHR: gl.COMPRESSED_SRGB8_ALPHA8_ASTC_8x5,
    COMPRESSED_SRGB8_ALPHA8_ASTC_8x6_KHR: gl.COMPRESSED_SRGB8_ALPHA8_ASTC_8x6,
    COMPRESSED_SRGB8_ALPHA8_ASTC_8x8_KHR: gl.COMPRESSED_SRGB8_ALPHA8_ASTC_8x8,
    COMPRESSED_SRGB8_ALPHA8_ASTC_10x5_KHR: gl.COMPRESSED_SRGB8_ALPHA8_ASTC_10x5,
    COMPRESSED_SRGB8_ALPHA8_ASTC_10x6_KHR: gl.COMPRESSED_SRGB8_ALPHA8_ASTC_10x6,
    COMPRESSED_SRGB8_ALPHA8_ASTC_10x8_KHR: gl.COMPRESSED_SRGB8_ALPHA8_ASTC_10x8,
    COMPRESSED_SRGB8_ALPHA8_ASTC_10x10_KHR: gl.COMPRESSED_SRGB8_ALPHA8_ASTC_10x10,
    COMPRESSED_SRGB8_ALPHA8_ASTC_12x10_KHR: gl.COMPRESSED_SRGB8_ALPHA8_ASTC_12x10,
    COMPRESSED_SRGB8_ALPHA8_ASTC_12x12_KHR: gl.COMPRESSED_SRGB8_ALPHA8_ASTC_12x12,
  },
  WEBGL_compressed_texture_atc: {
    COMPRESSED_RGB_ATC_WEBGL: gl.COMPRESSED_RGB_ATC_WEBGL,
    COMPRESSED_RGBA_ATC_EXPLICIT_ALPHA_WEBGL: gl.COMPRESSED_RGBA_ATC_EXPLICIT_ALPHA_WEBGL,
    COMPRESSED_RGBA_ATC_INTERPOLATED_ALPHA_WEBGL: gl.COMPRESSED_RGBA_ATC_INTERPOLATED_ALPHA_WEBGL,
  },
  WEBGL_compressed_texture_etc: {
    COMPRESSED_R11_EAC: gl.COMPRESSED_R11_EAC,
    COMPRESSED_SIGNED_R11_EAC: gl.COMPRESSED_SIGNED_R11_EAC,
    COMPRESSED_RG11_EAC: gl.COMPRESSED_RG11_EAC,
    COMPRESSED_SIGNED_RG11_EAC: gl.COMPRESSED_SIGNED_RG11_EAC,
    COMPRESSED_RGB8_ETC2: gl.COMPRESSED_RGB8_ETC2,
    COMPRESSED_RGBA8_ETC2_EAC: gl.COMPRESSED_RGBA8_ETC2_EAC,
    COMPRESSED_SRGB8_ETC2: gl.COMPRESSED_SRGB8_ETC2,
    COMPRESSED_SRGB8_ALPHA8_ETC2_EAC: gl.COMPRESSED_SRGB8_ALPHA8_ETC2_EAC,
    COMPRESSED_RGB8_PUNCHTHROUGH_ALPHA1_ETC2: gl.COMPRESSED_RGB8_PUNCHTHROUGH_ALPHA1_ETC2,
    COMPRESSED_SRGB8_PUNCHTHROUGH_ALPHA1_ETC2: gl.COMPRESSED_SRGB8_PUNCHTHROUGH_ALPHA1_ETC2,
  },
  WEBGL_compressed_texture_etc1: {
    COMPRESSED_RGB_ETC1_WEBGL: 0x8D64,
  },
  WEBGL_compressed_texture_pvrtc: {
    COMPRESSED_RGB_PVRTC_4BPPV1_IMG: 0x8C00,
    COMPRESSED_RGBA_PVRTC_4BPPV1_IMG: 0x8C02,
    COMPRESSED_RGB_PVRTC_2BPPV1_IMG: 0x8C01,
    COMPRESSED_RGBA_PVRTC_2BPPV1_IMG: 0x8C03,
  },
  WEBGL_compressed_texture_s3tc: {
    COMPRESSED_RGB_S3TC_DXT1_EXT: gl.COMPRESSED_RGB_S3TC_DXT1,
    COMPRESSED_RGBA_S3TC_DXT1_EXT: gl.COMPRESSED_RGBA_S3TC_DXT1,
    COMPRESSED_RGBA_S3TC_DXT3_EXT: gl.COMPRESSED_RGBA_S3TC_DXT3,
    COMPRESSED_RGBA_S3TC_DXT5_EXT: gl.COMPRESSED_RGBA_S3TC_DXT5,
  },
  WEBGL_compressed_texture_s3tc_srgb: {
    COMPRESSED_SRGB_S3TC_DXT1_EXT: gl.COMPRESSED_SRGB_S3TC_DXT1,
    COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT: gl.COMPRESSED_SRGB_ALPHA_S3TC_DXT1,
    COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT: gl.COMPRESSED_SRGB_ALPHA_S3TC_DXT3,
    COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT: gl.COMPRESSED_SRGB_ALPHA_S3TC_DXT5,
  },
  WEBGL_debug_renderer_info: {
    UNMASKED_VENDOR_WEBGL: gl.UNMASKED_VENDOR_WEBGL,
    UNMASKED_RENDERER_WEBGL: gl.UNMASKED_RENDERER_WEBGL,
  },
  WEBGL_debug_shaders: {getTranslatedShaderSource: gl.getTranslatedShaderSource || (() => {})},
  WEBGL_depth_texture: {
    UNSIGNED_INT_24_8_WEBGL: gl.UNSIGNED_INT_24_8,
  },
  WEBGL_draw_buffers: {
    COLOR_ATTACHMENT0_WEBGL: gl.COLOR_ATTACHMENT0,
    COLOR_ATTACHMENT1_WEBGL: gl.COLOR_ATTACHMENT1,
    COLOR_ATTACHMENT2_WEBGL: gl.COLOR_ATTACHMENT2,
    COLOR_ATTACHMENT3_WEBGL: gl.COLOR_ATTACHMENT3,
    COLOR_ATTACHMENT4_WEBGL: gl.COLOR_ATTACHMENT4,
    COLOR_ATTACHMENT5_WEBGL: gl.COLOR_ATTACHMENT5,
    COLOR_ATTACHMENT6_WEBGL: gl.COLOR_ATTACHMENT6,
    COLOR_ATTACHMENT7_WEBGL: gl.COLOR_ATTACHMENT7,
    COLOR_ATTACHMENT8_WEBGL: gl.COLOR_ATTACHMENT8,
    COLOR_ATTACHMENT9_WEBGL: gl.COLOR_ATTACHMENT9,
    COLOR_ATTACHMENT10_WEBGL: gl.COLOR_ATTACHMENT10,
    COLOR_ATTACHMENT11_WEBGL: gl.COLOR_ATTACHMENT11,
    COLOR_ATTACHMENT12_WEBGL: gl.COLOR_ATTACHMENT12,
    COLOR_ATTACHMENT13_WEBGL: gl.COLOR_ATTACHMENT13,
    COLOR_ATTACHMENT14_WEBGL: gl.COLOR_ATTACHMENT14,
    COLOR_ATTACHMENT15_WEBGL: gl.COLOR_ATTACHMENT15,
    DRAW_BUFFER0_WEBGL: gl.DRAW_BUFFER0,
    DRAW_BUFFER1_WEBGL: gl.DRAW_BUFFER1,
    DRAW_BUFFER2_WEBGL: gl.DRAW_BUFFER2,
    DRAW_BUFFER3_WEBGL: gl.DRAW_BUFFER3,
    DRAW_BUFFER4_WEBGL: gl.DRAW_BUFFER4,
    DRAW_BUFFER5_WEBGL: gl.DRAW_BUFFER5,
    DRAW_BUFFER6_WEBGL: gl.DRAW_BUFFER6,
    DRAW_BUFFER7_WEBGL: gl.DRAW_BUFFER7,
    DRAW_BUFFER8_WEBGL: gl.DRAW_BUFFER8,
    DRAW_BUFFER9_WEBGL: gl.DRAW_BUFFER9,
    DRAW_BUFFER10_WEBGL: gl.DRAW_BUFFER10,
    DRAW_BUFFER11_WEBGL: gl.DRAW_BUFFER11,
    DRAW_BUFFER12_WEBGL: gl.DRAW_BUFFER12,
    DRAW_BUFFER13_WEBGL: gl.DRAW_BUFFER13,
    DRAW_BUFFER14_WEBGL: gl.DRAW_BUFFER14,
    DRAW_BUFFER15_WEBGL: gl.DRAW_BUFFER15,
    MAX_COLOR_ATTACHMENTS_WEBGL: gl.MAX_COLOR_ATTACHMENTS,
    MAX_DRAW_BUFFERS_WEBGL: gl.MAX_DRAW_BUFFERS,
  },
  WEBGL_lose_context: {
    loseContext() {},
    restoreContext() {},
  },
  EXT_float_blend: {},
};
