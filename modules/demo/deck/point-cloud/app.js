/* eslint-disable no-unused-vars */
const {OrbitView, COORDINATE_SYSTEM, LinearInterpolator} = require('@deck.gl/core');
import DeckGL from '@deck.gl/react';
import {PointCloudLayer} from '@rapidsai/deck.gl';
import React, {PureComponent, useEffect} from 'react';
import {render} from 'react-dom';

const transitionInterpolator = new LinearInterpolator(['rotationOrbit']);

// import {IO} from '@rapidsai/io';
// let df = IO.readLas(`${__dirname}/../../../io/test/test.las`)
//            .select(['x', 'y', 'z'])
//            .castAll(new Float32);

import {DataFrame, Float32, Series, Uint32} from '@rapidsai/cudf';

const df = new DataFrame({
  'x': Series.new({
    type: new Float32,
    data: [
      0,         0,         31.41076,  19.58429,  -6.98955,  -28.30012, -28.30012,
      -6.98955,  19.58429,  31.41076,  62.79052,  55.59824,  35.66908,  7.56856,
      -22.26583, -46.99938, -60.96594, -60.96594, -46.99938, -22.26583, 7.56856,
      35.66908,  55.59824,  62.79052,  94.10831,  89.00927,  74.26468,  51.47237,
      23.10223,  -7.7714,   -37.80288, -63.73783, -82.76579
    ]
  }),
  'y': Series.new({
    type: new Float32,
    data: [
      0,         0,         0,        24.55792,  30.62323,  13.62862,  -13.62862,
      -30.62323, -24.55792, 0,        0,         29.18021,  51.67558,  62.33271,
      58.71016,  41.63782,  15.02675, -15.02675, -41.63782, -58.71016, -62.33271,
      -51.67558, -29.18021, 0,        0,         30.55692,  57.80252,  78.78433,
      91.22862,  93.78689,  86.18188, 69.23774,  44.79061
    ]
  }),
  'z': Series.new({
    type: new Float32,
    data: [
      0,       0,       10, 0.49344, 0.49344, 0.49344, 0.49344, 0.49344, 0.49344,
      0.49344, 1.97327, 5,  1.97327, 1.97327, 1.97327, 1.97327, 1.97327, 1.97327,
      1.97327, 1.97327, 11, 1.97327, 1.97327, 1.97327, 4.43804, 4.43804, 4.43804,
      4.43804, 4.43804, 19, 4.43804, 4.43804, 4.43804
    ]
  }),
  'color': Series.new({
    type: new Uint32,
    data: [
      4293326232, 4294815329, 4294208835, 4294967231, 4293326232, 4293326232, 4294815329,
      4294208835, 4294967231, 4293326232, 4293326232, 4294815329, 4294208835, 4294967231,
      4293326232, 4293326232, 4294815329, 4294208835, 4294967231, 4293326232, 4293326232,
      4294815329, 4294208835, 4294967231, 4293326232, 4293326232, 4294815329, 4294208835,
      4294967231, 4293326232, 4293326232, 4294815329, 4294208835
    ]
  })
})

console.log('Number of rendered points', df.numRows);

const INITIAL_VIEW_STATE = {
  target: [df.get('x').mean(), df.get('y').mean(), df.get('z').mean()],
  rotationX: 0,
  rotationOrbit: 0,
  maxZoom: 10,
  zoom: 1
};

export default class App extends PureComponent {
  constructor(props) {
    super(props);

    this.state = {viewState: INITIAL_VIEW_STATE};

    this._onViewStateChange = this._onViewStateChange.bind(this);
  }

  _onViewStateChange({viewState}) { this.setState({viewState}); }

  render() {
    const {viewState} = this.state;

    const layers = [new PointCloudLayer({
      id: 'laz-point-cloud-layer',
      numPoints: df.numRows,
      data: {
        points: {
          offset: 0,
          length: df.numRows,
          attributes: {
            pointPositionX: df.get('x').data,
            pointPositionY: df.get('y').data,
            pointPositionZ: df.get('z').data,
            pointColor: df.get('color').data,
          }
        }
      },
      coordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
      getNormal: [0, 1, 0],
      // getColor: [138, 245, 66, 255],
      opacity: 0.5,
      pointSize: 3,
      sizeUnits: 'pixels'
    })];

    return (
      <DeckGL
        views={new OrbitView({orbitAxis: 'Y', fov: 50, transitionInterpolator})}
        viewState={viewState}
        controller={true}
        onViewStateChange={this._onViewStateChange}
        layers={
      layers}
      //   parameters={
      // // { clearColor: [0.93, 0.86, 0.81, 1] }}
      />
    );
  }
}

export function renderToDOM(container) {
  render(<App />, container);
  }
