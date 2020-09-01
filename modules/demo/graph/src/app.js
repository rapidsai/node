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

import React from 'react';
import DeckGL from '@deck.gl/react';
import { GraphLayer } from './layers/graph';
import { OrthographicView } from '@deck.gl/core';

const loadGraphData = require(
    process.env.REACT_APP_ENVIRONMENT === 'browser'
        ? './services/triangle'
        : './services/remote'
).default;

import { as as asAsyncIterable } from 'ix/asynciterable/as';
import { takeWhile } from 'ix/asynciterable/operators/takewhile';

export class App extends React.Component {
    constructor(...args) {
        super(...args);
        this._isMounted = false;
        this.state = { graph: {}, autoCenter: true };
    }
    componentWillUnmount() {
        this._isMounted = false;
    }
    componentDidMount() {
        this._isMounted = true;
        asAsyncIterable(loadGraphData(this.props))
            .pipe(takeWhile(() => this._isMounted))
            .forEach((state) => this.setState(state));
    }
    render() {
        const props = { ...this.props };
        if (this.state.autoCenter && this.state.bbox) {
            const viewState = centerOnBbox(this.state.bbox);
            viewState && (props.initialViewState = viewState);
        }
        return (
            <DeckGL {...props}
                onAfterRender={this.state.onAfterRender}
                onViewStateChange={() => this.setState({ autoCenter: false })}>
                <GraphLayer
                    edgeOpacity={.1}
                    nodesStroked={true}
                    edgeStrokeWidth={1.5}
                    nodeFillOpacity={.1}
                    nodeStrokeOpacity={.9}
                    {...this.state.graph}
                    />
            </DeckGL>
        );
    }
}

export default App;

App.defaultProps = {
    controller: true,
    onWebGLInitialized,
    onHover: onDragEnd,
    onDrag: onDragStart,
    onDragEnd: onDragEnd,
    onDragStart: onDragStart,
    initialViewState: {
        minZoom: Number.NEGATIVE_INFINITY,
        maxZoom: Number.POSITIVE_INFINITY,
    },
    views: [
        new OrthographicView({
            clear: {
                color: [...[51, 51, 57].map((x) => x / 255), 1]
            }
        })
    ],
};

function onDragStart({ index }, { target }) {
    if (target) {
        [window, target].forEach((element) => (element.style || {}).cursor = 'grabbing');
    }
}

function onDragEnd({ index }, { target }) {
    if (target) {
        [window, target].forEach((element) => (element.style || {}).cursor = ~index ? 'pointer' :  'default');
    }
}

function onWebGLInitialized(gl) {
    if (gl.opengl) {
        gl.enable(gl.PROGRAM_POINT_SIZE);
        gl.enable(gl.POINT_SPRITE);
    }
}

function centerOnBbox([minX, maxX, minY, maxY]) {
    const width = maxX - minX, height = maxY - minY;
    if ((width === width) && (height === height)) {
        const world = width > height ? width : height;
        const screen = width > height ? window.outerWidth : window.outerHeight;
        const zoom = world > screen ? -(world / screen * 1.5) : (screen / world * .5);
        return {
            minZoom: Number.NEGATIVE_INFINITY,
            maxZoom: Number.POSITIVE_INFINITY,
            zoom: Math.log2(Math.abs(zoom)) * Math.sign(zoom),
            target: [minX + (width * .5), minY + (height * .5), 0],
        };
    }
}

import { log as deckLog } from '@deck.gl/core';
deckLog.level = 0;
