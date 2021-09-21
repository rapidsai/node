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

import {clamp} from 'math.gl';
import {DeckController} from '../deck.gl';

const {Controller} = require('@deck.gl/core');
const {OrbitState} = require('@deck.gl/core/dist/es5/controllers/orbit-controller');

class OrthographicState extends OrbitState {
  public declare zoomAxis: 'X'|'Y'|'all';

  constructor(props: any) {
    super(props);
    this.zoomAxis = props.zoomAxis || 'all';
  }

  _applyConstraints(props: any) {
    const {maxZoom, minZoom, zoom} = props;
    props.zoom                     = Array.isArray(zoom)
                                       ? [clamp(zoom[0], minZoom, maxZoom), clamp(zoom[1], minZoom, maxZoom)]
                                       : clamp(zoom, minZoom, maxZoom);
    return props;
  }

  _calculateNewZoom({scale, startZoom}: any) {
    const {_viewportProps}   = <any>this;
    const {maxZoom, minZoom} = _viewportProps;
    if (!startZoom && startZoom !== 0) { startZoom = _viewportProps.zoom; }
    let deltaZoom = Math.log2(scale);
    if (Array.isArray(startZoom)) {
      let [newZoomX, newZoomY] = startZoom;
      switch (this.zoomAxis) {
        case 'X':
          // Scale x only
          newZoomX = clamp(newZoomX + deltaZoom, minZoom, maxZoom);
          break;
        case 'Y':
          // Scale y only
          newZoomY = clamp(newZoomY + deltaZoom, minZoom, maxZoom);
          break;
        default:
          // Lock aspect ratio
          let z = Math.min(newZoomX + deltaZoom, newZoomY + deltaZoom);
          if (z < minZoom) { deltaZoom += minZoom - z; }
          z = Math.max(newZoomX + deltaZoom, newZoomY + deltaZoom);
          if (z > maxZoom) { deltaZoom += maxZoom - z; }
          newZoomX += deltaZoom;
          newZoomY += deltaZoom;
      }
      return [newZoomX, newZoomY];
    }
    // Ignore `zoomAxis`
    // `LinearTransitionInterpolator` does not support interpolation between a number and an array
    // So if zoom is a number (legacy use case), new zoom still has to be a number
    return clamp(startZoom + deltaZoom, minZoom, maxZoom);
  }

  _getUpdatedState(newProps: any) {
    // Update _viewportProps
    return new (this.constructor as any)({...this._viewportProps, ...this._state, ...newProps});
  }
}

export default class OrthographicController extends (Controller as typeof DeckController) {
  protected declare _linearTransitionProps: any;

  constructor(props: any) {
    props.dragMode = props.dragMode || 'pan';
    super(OrthographicState, props);
    this._linearTransitionProps = props.linearTransitionProps || null;
  }

  _onPanRotate(_event: any) {
    // No rotation in orthographic view
    return false;
  }

  get linearTransitionProps() { return this._linearTransitionProps; }
}
