// Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

import { log as deckLog, OrthographicView } from '@deck.gl/core';
import { TextLayer } from '@deck.gl/layers';
import { default as DeckGL } from '@deck.gl/react';
import { GraphLayer } from '@rapidsai/deck.gl';
import { as as asAsyncIterable } from 'ix/asynciterable/as';
import { takeWhile } from 'ix/asynciterable/operators/takewhile';
import * as React from 'react';

import { default as loadGraphData } from './loader';

deckLog.level = 0;
deckLog.enable(false);

const composeFns = (fns) => function (...args) { fns.forEach((fn) => fn && fn.apply(this, args)); }

export class App extends React.Component {
  constructor(props, context) {
    super(props, context);
    this._isMounted = false;
    this._deck = React.createRef();
    this.state = { graph: {}, autoCenter: true };
  }
  componentWillUnmount() { this._isMounted = false; }
  componentDidMount() {
    this._isMounted = true;
    asAsyncIterable(loadGraphData(this.props))
      .pipe(takeWhile(() => this._isMounted))
      .forEach((state) => this.setState(state))
      .catch((e) => console.error(e));
  }
  render() {
    const { onAfterRender, ...props } = this.props;
    const { params = {}, selectedParameter } = this.state;
    const [viewport] = (this._deck?.current?.deck?.viewManager?.getViewports() || []);

    if (this.state.autoCenter && this.state.bbox) {
      const viewState = centerOnBbox(this.state.bbox);
      viewState && (props.initialViewState = viewState);
    }

    const [minX = Number.NEGATIVE_INFINITY, minY = Number.NEGATIVE_INFINITY] =
      (viewport?.getBounds() || []);

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
          getNodeLabels={getNodeLabels}
          getEdgeLabels={getEdgeLabels}
        />
        {viewport && selectedParameter !== undefined ?
          <TextLayer
            sizeScale={1}
            opacity={0.9}
            maxWidth={2000}
            pickable={false}
            background={true}
            getBackgroundColor={() => [46, 46, 46]}
            getTextAnchor='start'
            getAlignmentBaseline='top'
            getSize={(d) => d.size}
            getColor={(d) => d.color}
            getPixelOffset={(d) => [0, d.index * 15]}
            getPosition={(d) => d.position}
            data={Object.keys(params).map((key, i) => ({
              size: 15,
              index: i,
              text: i === selectedParameter
                ? `(${i}) ${params[key].name}: ${params[key].val}`
                : ` ${i}  ${params[key].name}: ${params[key].val}`,
              color: [255, 255, 255],
              position: [minX, minY],
            }))}
          /> : null
        }
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
  if (target) { [window, target].forEach((element) => (element.style || {}).cursor = 'grabbing'); }
}

function onDragEnd({ index }, { target }) {
  if (target) {
    [window, target].forEach((element) => (element.style || {}).cursor =
      ~index ? 'pointer' : 'default');
  }
}

function centerOnBbox([minX, maxX, minY, maxY]) {
  const width = maxX - minX, height = maxY - minY;
  if ((width === width) && (height === height)) {
    const { outerWidth, outerHeight } = window;
    const world = (width > height ? width : height);
    const screen = (width > height ? outerWidth : outerHeight) * .9;
    const zoom = (world > screen ? -(world / screen) : (screen / world));
    return {
      minZoom: Number.NEGATIVE_INFINITY,
      maxZoom: Number.POSITIVE_INFINITY,
      zoom: Math.log2(Math.abs(zoom)) * Math.sign(zoom),
      target: [minX + (width * .5), minY + (height * .5), 0],
    };
  }
}

function getNodeLabels({ x, y, coordinate, nodeId, props, layer }) {
  let size = 14;
  const color = [255, 255, 255];
  const labels = [{
    size,
    color,
    offset: [14, 0],
    position: () => {
      const x = window.mouseInWindow ? window.mouseX : -1;
      const y = window.mouseInWindow ? window.mouseY : -1;
      return layer.context.viewport.unproject([x, y]);
    },
    text: props.data.nodes.attributes.nodeName
      ? props.data.nodes.attributes.nodeName.getValue(nodeId)
      : `${nodeId}`,
  }];
  if (props.data.nodes.attributes.nodeData) {
    size *= 1.5;
    const text = `${props.data.nodes.attributes.nodeData.getValue(nodeId) || ''}`.trimEnd();
    const lineSpacing = 3, padding = 20;
    const nBreaks = ((text.match(/\n/ig) || []).length + 1);
    const offsetX = 0, offsetY = (size + lineSpacing) * nBreaks;
    labels.push({
      text,
      color,
      size,
      position: () => {
        const [minX, , , maxY] = layer.context.viewport.getBounds();
        return [minX, maxY];
      },
      offset: [offsetX + padding, -padding - offsetY],
    });
  }
  return labels;
}

function getEdgeLabels({ x, y, coordinate, edgeId, props, layer }) {
  let size = 14;
  const color = [255, 255, 255];
  const labels = [{
    size,
    color,
    offset: [14, 0],
    position: () => {
      const x = window.mouseInWindow ? window.mouseX : -1;
      const y = window.mouseInWindow ? window.mouseY : -1;
      return layer.context.viewport.unproject([x, y]);
    },
    text: props.data.edges.attributes.edgeName
      ? props.data.edges.attributes.edgeName.getValue(edgeId)
      : `${sourceNodeId} - ${targetNodeId}`,
  }];
  if (props.data.edges.attributes.edgeData) {
    size *= 1.5;
    const text = `${props.data.edges.attributes.edgeData.getValue(edgeId) || ''}`.trimEnd();
    const lineSpacing = 3, padding = 20;
    const nBreaks = ((text.match(/\n/ig) || []).length + 1);
    const offsetX = 0, offsetY = (size + lineSpacing) * nBreaks;
    labels.push({
      text,
      color,
      size,
      offset: [offsetX + padding, -padding - offsetY],
      position: () => {
        const [minX, , , maxY] = layer.context.viewport.getBounds();
        return [minX, maxY];
      },
    });
  }
  return labels;
}
