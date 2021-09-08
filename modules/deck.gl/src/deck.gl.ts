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

declare module '@deck.gl/core' {
  export class Deck {
    [key: string]: any;
  }

  export interface DeckContext {
    gl: WebGL2RenderingContext;
    shaderCache: any;
  }
  export interface UpdateStateProps {
    oldProps: any;
    props: any;
    context: DeckContext;
    changeFlags: any;
  }
  export interface PickingInfo {
    /** Mouse position x relative to the viewport. */
    x: number;
    /** Mouse position y relative to the viewport. */
    y: number;
    /**
       Mouse position in world coordinates. Only applies if coordinateSystem is
       COORDINATE_SYSTEM.LNGLAT.
     */
    coordinate: [number, number];
    /**
       The color of the pixel that is being picked. It represents a "picking color" that is encoded
       by layer.encodePickingColor().
     */
    color: [number, number, number, number];
    /**
       The index of the object that is being picked. It is the returned value of
       layer.decodePickingColor().
     */
    index: number;
    /** true if index is not -1. */
    picked: boolean;

    [key: string]: any;
  }

  export const picking: any;
  export const project32: any;

  export class Layer {
    constructor(props?: any);
    props: any;
    state: any;
    internalState: any;
    setState(updatedState: any): void;
    initializeState(context: DeckContext): void;
    shouldUpdateState(options: UpdateStateProps): boolean;
    updateState(options: UpdateStateProps): void;
    finalizeState(context?: DeckContext): void;
    draw(options: {moduleParameters: any, uniforms: any, context: DeckContext}): void;
    getPickingInfo({info, mode}: {info: PickingInfo, mode: 'hover'|'click'}): void;
    onHover(info: PickingInfo, pickingEvent: any): void;
    getAttributeManager(): any;
  }
  export class CompositeLayer extends Layer {
    getSubLayerProps(subLayerProps: any): any;
  }
}

declare module '@deck.gl/layers' {
import {Layer, CompositeLayer, picking, project32} from '@deck.gl/core';
  export {Layer, CompositeLayer, picking, project32};
  export class TextLayer extends Layer {
    constructor(props?: any);
  }
}
