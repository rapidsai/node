import { OrthographicView } from '@deck.gl/core';
import DeckGL from '@deck.gl/react';
import * as React from 'react';
import { Component } from 'react';
import * as ReactDOM from 'react-dom';

import BezierGraphLayer from './bezier-graph-layer';
import SAMPLE_GRAPH from './sample-graph.json';

const INITIAL_VIEW_STATE = {
  target: [0, 0, 0],
  zoom: 1
};

export default class App extends Component {
  render() {
    const { data = SAMPLE_GRAPH, ...props } = this.props;

    return (
      <DeckGL
        width='100%'
        height='100%'
        initialViewState={INITIAL_VIEW_STATE}
        controller={true}
        views={new OrthographicView()}
        layers={[new BezierGraphLayer({ data })]}
        {...props}
      />
    );
  }
}

if (require.main === module) {
  ReactDOM.render(<App />, document.body.appendChild(document.createElement('div')));
}
