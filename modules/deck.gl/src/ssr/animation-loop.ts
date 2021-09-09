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

import {AnimationLoop as DeckAnimationLoop} from '@luma.gl/engine';
import {setParameters} from '@luma.gl/gltools';
import {cancelAnimationFrame, requestAnimationFrame} from '@luma.gl/webgl';

const {
  start,
  _getPageLoadPromise,
  _initializeCallbackData,
} = <any>DeckAnimationLoop.prototype;

export class AnimationLoop extends DeckAnimationLoop {
  declare private _sync: boolean;
  declare private _running: boolean;
  declare private _initialized: boolean;
  declare private _animationProps: any;
  declare private _pageLoadPromise: any;
  declare private _animationFrameId: any;
  constructor(props = {}) {
    super(props);
    this._sync           = (<any>props)._sync || false;
    this._animationProps = (<any>props).animationProps;
  }
  start(opts = {}): DeckAnimationLoop {
    if (this._initialized && !this._running) {
      this._running = true;
      (<any>this)._startLoop();
      return this;
    }
    return start.call(this, opts);
  }
  pause() {
    if (this._running) {
      this._running = false;
      if (this._animationFrameId) { cancelAnimationFrame(this._animationFrameId); }
    }
    return this;
  }
  onBeforeRender(animationProps: any) {
    const {_onBeforeRender} = (<any>this.props);
    _onBeforeRender && _onBeforeRender(animationProps);
  }
  _renderFrame(animationProps: any) {
    if (this.framebuffer) {  //
      setParameters(this.gl, {framebuffer: this.framebuffer});
    }
    this.onBeforeRender(animationProps);
    this.onRender(animationProps);
    this.onAfterRender(animationProps);
  }
  onAfterRender(animationProps: any) {
    const {_onAfterRender} = (<any>this.props);
    _onAfterRender && _onAfterRender(animationProps);
  }
  _getPageLoadPromise() {
    if (this._sync) {
      if (!this._pageLoadPromise) {
        this._pageLoadPromise =
          new ImmediatePromise(typeof document !== 'undefined' ? document : {});
      }
      return this._pageLoadPromise;
    }
    return _getPageLoadPromise.call(this);
  }
  _requestAnimationFrame(renderFrameCallback: any) {
    return this._running ? requestAnimationFrame(renderFrameCallback) : undefined;
  }
  _initializeCallbackData() {
    const animationProps = _initializeCallbackData.call(this);
    if (this._animationProps) {
      const {
        useDevicePixels = animationProps.useDevicePixels,
        needsRedraw     = animationProps.needsRedraw,
        startTime       = animationProps.startTime,
        engineTime      = animationProps.engineTime,
        width           = animationProps.width,
        height          = animationProps.height,
        aspect          = animationProps.aspect,
        tick            = animationProps.tick,
        tock            = animationProps.tock,
        time            = animationProps.time,
        _mousePosition  = animationProps._mousePosition,
      } = this._animationProps;

      animationProps.useDevicePixels = useDevicePixels;
      animationProps.needsRedraw     = needsRedraw;
      animationProps.startTime       = startTime;
      animationProps.engineTime      = engineTime;
      animationProps.width           = width;
      animationProps.height          = height;
      animationProps.aspect          = aspect;
      animationProps.tick            = tick;
      animationProps.tock            = tock;
      animationProps.time            = time;
      animationProps._mousePosition  = _mousePosition;
    }
    return animationProps;
  }
}

class ImmediatePromise {
  declare private _value: any;
  declare private _error: any;
  constructor(value: any = undefined, error: any = undefined) {
    this._value = value;
    this._error = error;
  }
  catch(onError?: (error: any) => any) {
    if (this._error) {
      try {
        if (typeof onError === 'function') {
          const x = onError(this._error);
          return (typeof x?.then === 'function') ? x : new ImmediatePromise(x);
        }
      } catch (e1) { return new ImmediatePromise(undefined, e1); }
    }
    return this;
  }
  then(onValue?: (value: any) => any, onError?: (error: any) => any) {
    try {
      if (typeof onValue === 'function') {
        const x = onValue(this._value);
        return (typeof x?.then === 'function') ? x : new ImmediatePromise(x);
      }
    } catch (e1) {
      if (typeof onError === 'function') {
        try {
          const x = onError(e1);
          return (typeof x?.then === 'function') ? x : new ImmediatePromise(x);
        } catch (e2) { return new ImmediatePromise(undefined, e2); }
      } else {
        return new ImmediatePromise(undefined, e1);
      }
    }
    return this;
  }
}
