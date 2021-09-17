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
      onBeforeAnimationFrameRender,
      onAfterAnimationFrameRender,
      animationProps,
      useDevicePixels,
      createFramebuffer,
      autoResizeDrawingBuffer
    } = props;

    const getFramebufferFromLoop = createFramebuffer && !props._framebuffer;

    const loop = new AnimationLoop({
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
      onError,
      _onBeforeRender: onBeforeAnimationFrameRender,
      _onAfterRender: onAfterAnimationFrameRender,
    });

    if (getFramebufferFromLoop) {
      const {_createFramebuffer}     = <any>loop;
      (<any>loop)._createFramebuffer = () => {
        _createFramebuffer.call(loop);
        this.setProps({_framebuffer: loop.framebuffer});
      };
    }

    return loop;
  }
  _onViewStateChange(params: any) {
    this._latestViewState = {...params.viewState};
    // if (this._latestViewState.minZoom == null) {
    //   this._latestViewState.minZoom = Number.NEGATIVE_INFINITY;
    // }
    // if (this._latestViewState.maxZoom == null) {
    //   this._latestViewState.maxZoom = Number.POSITIVE_INFINITY;
    // }
    // Object.assign(params.viewState, this._latestViewState);
    // Object.assign(this.interactiveState, params.interactionState);
    // const viewState = {...params.viewState};
    super._onViewStateChange(params);
    // console.log({
    //   viewState1: viewState,
    //   viewState2: this.viewState,
    // });
  }
  _onInteractionStateChange(interactionState: any) {
    return super._onInteractionStateChange(Object.assign(this.interactiveState, interactionState));
  }
  restore(state: any) {
    if (state.props) { this.setProps(state.props); }
    if ('width' in state) { this.width = state.width; }
    if ('height' in state) { this.height = state.height; }
    if (state.metrics) { this.metrics = {...this.metrics, ...state.metrics}; }
    if ('_metricsCounter' in state) { this._metricsCounter = state._metricsCounter; }
    if (state._pickRequest) { this._pickRequest = {...this._pickRequest, ...state._pickRequest}; }
    if (state.props?.initialViewState) { this._latestViewState = state.props.initialViewState; }
    if (state.interactiveState) {
      this.interactiveState = {...this.interactiveState, ...state.interactiveState};
    }
    if (state._lastPointerDownInfo) {
      this._lastPointerDownInfo = {...this._lastPointerDownInfo, ...state._lastPointerDownInfo};
    }

    if ('animationProps' in state && this.animationLoop) {
      this.animationLoop.restore(state.animationProps);
    }

    if (state.views && this.viewManager) {
      const viewMap = this.viewManager.getViews();
      [...Object.keys(viewMap), ...Object.keys(state.views)]
        .map((viewId) => [state.views[viewId], viewMap[viewId]?.props])
        .filter(([source, target]) => source && target)
        .forEach(([source, target]) => Object.assign(target, source));
    }

    if (state.controllers && this.viewManager) {
      const controllersMap = this.viewManager.controllers;
      [...Object.keys(controllersMap), ...Object.keys(state.controllers)]
        .map((viewId) => [state.controllers[viewId], controllersMap[viewId]])
        .filter(([source, target]) => source && target)
        .forEach(([source, target]) => Object.assign(target, source));
    }

    if (state.deckPicker && this.deckPicker) {
      this.deckPicker.lastPickedInfo = {
        ...this.deckPicker.lastPickedInfo,
        index: state.deckPicker.lastPickedInfo?.index,
        layerId: state.deckPicker.lastPickedInfo?.layerId,
        info: {
          ...this.deckPicker.lastPickedInfo?.info,
          layer: {
            ...this.deckPicker.lastPickedInfo?.info?.layer,
            ...state.deckPicker.lastPickedInfo?.info?.layer,
          },
          viewport: {
            ...this.deckPicker.lastPickedInfo?.info?.viewport,
            ...state.deckPicker.lastPickedInfo?.info?.viewport,
          },
        }
      };
    }

    return this;
  }
  serialize() {
    // const {minZoom, maxZoom, target} = this._getViewState();
    // const viewState                  = {
    //   target,
    //   minZoom: minZoom !== null ? minZoom : Number.NEGATIVE_INFINITY,
    //   maxZoom: maxZoom !== null ? maxZoom : Number.POSITIVE_INFINITY,
    // };

    // const views = (() => {
    //   const views   = {} as any;
    //   const viewMap = this.viewManager.getViews();
    //   for (const viewId in viewMap) {  //
    //     views[viewId] = {controller: viewMap[viewId].controller};
    //   }
    //   return views;
    // })();

    // const controllers = (() => {
    //   const controllers = {} as any;
    //   for (const viewId in this.viewManager.controllers) {  //
    //     const controller    = this.viewManager.controllers[viewId];
    //     controllers[viewId] = {
    //       _state: controller._state,
    //       dragPan: controller.dragPan,
    //       keyboard: controller.keyboard,
    //       touchZoom: controller.touchZoom,
    //       dragRotate: controller.dragRotate,
    //       scrollZoom: controller.scrollZoom,
    //       touchRotate: controller.touchRotate,
    //       doubleClickZoom: controller.doubleClickZoom,
    //       _interactionState: controller._interactionState,
    //       _eventStartBlocked: controller._eventStartBlocked,
    //       controllerStateProps: controller.controllerStateProps,
    //     };
    //   }
    //   return controllers;
    // })();

    // if (this._latestViewState) { console.log(this._latestViewState); }

    return {
      // views,
      // controllers,
      width: this.width,
      height: this.height,
      _metricsCounter: this._metricsCounter,
      metrics: this.metrics ? {...this.metrics} : undefined,
      _pickRequest: this._pickRequest ? {...this._pickRequest} : undefined,
      animationProps: this.animationLoop ? this.animationLoop.serialize() : undefined,
      interactiveState: this.interactiveState ? {...this.interactiveState} : undefined,
      _lastPointerDownInfo: serializePickingInfo(this._lastPointerDownInfo),
      deckPicker: this.deckPicker.lastPickedInfo && {
        lastPickedInfo: serializeLastPickedInfo(this),
      },
      // viewState: {...this._getViewState()},
      props: {
        debug: this.props.debug,
        _animate: this.props._animate,
        _pickable: this.props._pickable,
        touchAction: this.props.touchAction,
        // initialViewState: {...this._latestViewState},
        // initialViewState: {...this._getViewState()},
        // initialViewState: {...this.props.initialViewState},
        useDevicePixels: this.props.useDevicePixels,
        drawPickingColors: this.props.drawPickingColors,
        _typedArrayManagerProps: this.props._typedArrayManagerProps,
      },
    };
  }
}

function serializePickingInfo(info: any) {
  const {layer, viewport, sourceLayer} = info || {};
  return info && {
    x: info.x,
    y: info.y,
    index: info.index,
    color: info.color,
    picked: info.picked,
    coordinate: info.coordinate,
    devicePixel: info.devicePixel,
    layer: layer && {id: layer.id},
    viewport: viewport && {id: viewport.id},
    sourceLayer: sourceLayer && {id: sourceLayer.id},
  };
}

function serializeLastPickedInfo({deckPicker}: any) {
  const {lastPickedInfo}       = deckPicker || {};
  const {index, layerId, info} = lastPickedInfo || {};
  return lastPickedInfo && {
    index,
    layerId,
    info: serializePickingInfo(info),
  };
}
