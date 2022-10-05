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

import { Series } from '@rapidsai/cudf';
import { GraphLayer } from '@rapidsai/deck.gl';

// Preload cuDF kernels into device memory
Series.new([0, 1, 2]).sum();

import * as React from 'react';
import DeckGL from '@deck.gl/react';
import { log as deckLog, OrthographicView } from '@deck.gl/core';

deckLog.level = 0;
deckLog.enable(false);

import { PolygonLayer, ScatterplotLayer, TextLayer } from '@deck.gl/layers';

import { ColorMapper } from './color';
import {
  loadPointsInCensusTracts,
  // loadPointsNearEachCensusTract,
} from './util';

export default class App extends React.Component {
  constructor(...args) {

    super(...args);

    this.state = {
      pointsByLevel: [],
      tract_polygons: [],
      tract_vertices: [],
    };

    this._deck = React.createRef();
    this._onViewStateChanged = this._onViewStateChanged.bind(this);

    loadPointsInCensusTracts(new ColorMapper())
      // loadPointsNearEachCensusTract(new ColorMapper())
      .then(({ bbox, pointsByLevel, tract_polygons, tract_vertices }) => {

        const { width, height } = bbox;
        const zoom = (() => {
          const outerWidth = window.outerWidth * .75;
          const outerHeight = window.outerHeight * .75;
          const world = (width > height ? width : height);
          const screen = (width > height ? outerWidth : outerHeight);
          const zoom = (world > screen ? -(world / screen) : (screen / world)) * 1.1;
          return Math.log2(Math.abs(zoom)) * Math.sign(zoom);
        })();

        this.setState({
          bbox,
          selectedLevels: [],
          highlightedLevel: undefined,
          rerender: true,
          pointsByLevel,
          tract_polygons,
          tract_vertices,
          viewState: {
            zoom,
            width: window.outerWidth,
            height: window.outerHeight,
            minZoom: Number.NEGATIVE_INFINITY,
            maxZoom: Number.POSITIVE_INFINITY,
            target: [
              bbox.xMin + width * 0.5,
              bbox.yMin + height * 0.5,
              0
            ],
          }
        });
      });
  }

  _onViewStateChanged({ viewState, oldViewState }) {
    this.setState({ viewState, rerender: viewState.zoom !== oldViewState.zoom, });
  }

  render() {

    let { xMin, yMin } = this.state.bbox || {};
    const [viewport] = this._deck?.current?.deck?.viewManager?.getViewports() || [];

    if (viewport) {
      const { target, width, height } = this.state.viewState;
      const [xMid, yMid] = viewport.project(target);
      [xMin, yMin] = viewport.unproject([
        xMid - (width * 0.5), yMid - (height * 0.5)
      ]);
    }

    const visibleLevels = [].concat(this.state.highlightedLevel ?? [], this.state.selectedLevels);

    return (
      <DeckGL
        ref={this._deck}
        viewState={this.state.viewState || {}}
        onViewStateChange={this._onViewStateChanged}
        controller={{ keyboard: false, doubleClickZoom: false }}
        onAfterRender={() => this.state.rerender && this.setState({ rerender: false })}
        onClick={() => this.setState({ selectedLevels: [], highlightedLevel: undefined })}
        views={[new OrthographicView({ clear: { color: [...[46, 46, 46].map((x) => x / 255), 1] } })]}
      >
        <PolygonLayer
          id='tract_polygons'
          opacity={0.5}
          filled={true}
          stroked={true}
          extruded={false}
          positionFormat={`XY`}
          lineWidthMinPixels={1}
          data={this.state.tract_polygons || []}
          getPolygon={({ rings }) => rings}
          getElevation={({ id }) => id}
          getFillColor={({ color }) => [...color.slice(0, 3), 15]}
          getLineColor={({ color }) => [...color.slice(0, 3), 255]}
        />
        <ScatterplotLayer
          id='tract_vertices'
          opacity={0.05}
          filled={false}
          stroked={true}
          radiusMinPixels={2}
          data={this.state.tract_vertices || []}
          getRadius={(x) => x.radius}
          getLineColor={(x) => x.color}
          getPosition={(x) => x.position}
          getLineWidth={(x) => x.strokeWidth}
        />
        <React.Fragment>
          {
            this.state.pointsByLevel
              .filter(({ level }) => visibleLevels.indexOf(level) !== -1)
              .map(({ numNodes, data, level }) => (
                <GraphLayer
                  id={`points_by_level_${level}`}
                  autoHighlight={false}
                  pickable={false}
                  data={data}
                  dataTransform={(nodes) => ({ edges: {}, nodes })}
                  numNodes={numNodes}
                  numEdges={0}
                  nodesVisible={true}
                  edgesVisible={false}
                  nodesFilled={true}
                  nodesStroked={false}
                  nodeRadiusScale={10}
                  nodeRadiusMinPixels={0}
                  nodeRadiusMaxPixels={1}
                />
              ))
          }
        </React.Fragment>
        <TextLayer
          data={this.state.pointsByLevel.map(({ level }) => ({
            level,
            size: 15,
            text: `Level ${level}`,
            position: [xMin, yMin],
            color: [255, 255, 255],
          }))}
          opacity={1.0}
          sizeScale={1}
          maxWidth={2000}
          pickable={true}
          background={true}
          getTextAnchor={'start'}
          getAlignmentBaseline={'top'}
          getSize={({ size }) => size}
          getColor={({ color, level }) => [...color, visibleLevels.indexOf(level) !== -1 ? 255 : 150]}
          getPosition={({ position }) => position}
          getPixelOffset={({ level }) => [0, level * 15]}
          getBackgroundColor={() => [46, 46, 46]}
          onHover={(info) => {
            this.setState({ highlightedLevel: info?.object?.level });
            return true;
          }}
          onClick={(info, event) => {
            const level = info?.object?.level;
            if (typeof level === 'number') {
              const { shiftKey = false } = (event.pointers && event.pointers[0] || {});
              if (this.state.selectedLevels.indexOf(level) === -1) {
                this.setState({
                  highlightedLevel: undefined,
                  selectedLevels: shiftKey ? this.state.selectedLevels.concat(level) : [level],
                });
              } else {
                this.setState({
                  highlightedLevel: shiftKey ? undefined : this.state.highlightedLevel,
                  selectedLevels: shiftKey ? this.state.selectedLevels.filter((l) => l !== level) : [level],
                });
              }
            }
            return true;
          }}
        />
      </DeckGL>
    );
  }
}
