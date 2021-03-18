// Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

import { log as deckLog } from '@deck.gl/core';

deckLog.level = 0;
deckLog.enable(false);

import { OrthographicView } from '@deck.gl/core';
import { TextLayer } from '@deck.gl/layers';
import DeckGL from '@deck.gl/react';
import { createDeckGLReactRef } from '@rapidsai/deck.gl';
import { as as asAsyncIterable } from 'ix/asynciterable/as';
import { takeWhile } from 'ix/asynciterable/operators/takewhile';
import React from 'react';

import { GraphLayer } from './layers/graph';

const composeFns = (fns) => function (...args) { fns.forEach((fn) => fn && fn.apply(this, args)); }

export class App extends React.Component {
  constructor(props, context) {
    if (props.serverRendered) { Object.assign(props, createDeckGLReactRef()); }
    super(props, context);
    this._isMounted = false;
    this._deck = React.createRef();
    this.state = { graph: {}, autoCenter: true };
  }
  componentWillUnmount() { this._isMounted = false; }
  componentDidMount() {
    this._isMounted = true;
    const loadGraphData = require(
      process.env.REACT_APP_ENVIRONMENT !== 'browser'
        ? this.props.url
          ? './services/remote'
          : './services/local'
        : './services/triangle'
    ).default;

    asAsyncIterable(loadGraphData(this.props))
      .pipe(takeWhile(() => this._isMounted))
      .forEach((state) => this.setState(state));
  }
  render() {
    const { onAfterRender, ...props } = this.props;
    const { params = {}, selectedParameter } = this.state;
    if (this.state.autoCenter && this.state.bbox) {
      const viewState = centerOnBbox(this.state.bbox);
      viewState && (props.initialViewState = viewState);
    }
    return (
      <DeckGL {...props}
        ref={this._deck}
        onViewStateChange={() => this.setState({
          autoCenter: params.autoCenter ? (params.autoCenter.val = false) : false
        })}
        _framebuffer={props.getRenderTarget ? props.getRenderTarget() : null}
        onAfterRender={composeFns([onAfterRender, this.state.onAfterRender])}>
        <GraphLayer
          edgeStrokeWidth={2}
          edgeOpacity={.5}
          nodesStroked={true}
          nodeFillOpacity={.5}
          nodeStrokeOpacity={.9}
          {...this.state.graph}
        />
        {selectedParameter !== undefined ?
          <TextLayer
            sizeScale={1}
            opacity={0.9}
            maxWidth={2000}
            pickable={false}
            backgroundColor={[46, 46, 46]}
            getTextAnchor='start'
            getAlignmentBaseline='top'
            getSize={(d) => d.size}
            getColor={(d) => d.color}
            getPixelOffset={(d) => [0, 0]}
            getPosition={(d) => this._deck.current.viewports[0].unproject(d.position)}
            data={Object.keys(params).map((key, i) => ({
              size: 15,
              text: i === selectedParameter
                ? `(${i}) ${params[key].name}: ${params[key].val}`
                : ` ${i}  ${params[key].name}: ${params[key].val}`,
              color: [255, 255, 255],
              position: [0, i * 15],
            }))}
          /> : null}
      </DeckGL>
    );
  }
}

export default App;

App.defaultProps = {
  controller: { keyboard: false },
  onHover: onDragEnd,
  onDrag: onDragStart,
  onDragEnd: onDragEnd,
  onDragStart: onDragStart,
  initialViewState: {
    zoom: 1,
    target: [0, 0, 0],
    minZoom: Number.NEGATIVE_INFINITY,
    maxZoom: Number.POSITIVE_INFINITY,
  },
  views: [
    new OrthographicView({
      clear: {
        color: [...[46, 46, 46].map((x) => x / 255), 1]
      }
    })
  ]
};

function onDragStart({ index }, { target }) {
  if (target) {
    [window, target].forEach((element) => (element.style || {}).cursor = 'grabbing');
  }
}

function onDragEnd({ index }, { target }) {
  if (target) {
    [window, target].forEach((element) =>
      (element.style || {}).cursor = ~index ? 'pointer' : 'default');
  }
}

function centerOnBbox([minX, maxX, minY, maxY]) {
  const width = maxX - minX, height = maxY - minY;
  if ((width === width) && (height === height)) {
    const { outerWidth, outerHeight } = window;
    const world = outerWidth > outerHeight ? width : height;
    const screen = outerWidth > outerHeight ? outerWidth : outerHeight;
    const zoom = world > screen ? -(world / screen) : (screen / world);
    // const zoom = world > screen ? -(world / screen * 1.5) : (screen / world * .5);
    return {
      minZoom: Number.NEGATIVE_INFINITY,
      maxZoom: Number.POSITIVE_INFINITY,
      zoom: Math.log2(Math.abs(zoom)) * Math.sign(zoom),
      target: [minX + (width * .5), minY + (height * .5), 0],
    };
  }
}
