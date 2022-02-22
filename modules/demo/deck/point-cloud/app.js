/* eslint-disable no-unused-vars */
import {COORDINATE_SYSTEM, LinearInterpolator, OrbitView} from '@deck.gl/core';
import DeckGL from '@deck.gl/react';
import {DataFrame, Float32, Float64, Int32, Series} from '@rapidsai/cudf';
import {PointCloudLayer} from '@rapidsai/deck.gl';
// import {load, registerLoaders} from '@loaders.gl/core';
// import {LASLoader} from '@loaders.gl/las';
// import {PLYLoader} from '@loaders.gl/ply';
import React, {PureComponent} from 'react';
import {render} from 'react-dom';

// Additional format support can be added here, see
// https://github.com/uber-web/loaders.gl/blob/master/docs/api-reference/core/register-loaders.md
// registerLoaders(LASLoader);
// registerLoaders(PLYLoader);

// // Data source: kaarta.com
// const LAZ_SAMPLE =
//   'https://raw.githubusercontent.com/uber-common/deck.gl-data/master/examples/point-cloud-laz/indoor.0.1.laz';
// // Data source: The Stanford 3D Scanning Repository
// const PLY_SAMPLE =
//   'https://raw.githubusercontent.com/uber-common/deck.gl-data/master/examples/point-cloud-ply/lucy800k.ply';

const df = new DataFrame({
  'x': [
    0,         0,         31.41076,  19.58429,  -6.98955,  -28.30012, -28.30012,
    -6.98955,  19.58429,  31.41076,  62.79052,  55.59824,  35.66908,  7.56856,
    -22.26583, -46.99938, -60.96594, -60.96594, -46.99938, -22.26583, 7.56856,
    35.66908,  55.59824,  62.79052,  94.10831,  89.00927,  74.26468,  51.47237,
    23.10223,  -7.7714,   -37.80288, -63.73783, -82.76579
  ],
  'y': [
    0,         0,         0,        24.55792,  30.62323,  13.62862,  -13.62862,
    -30.62323, -24.55792, 0,        0,         29.18021,  51.67558,  62.33271,
    58.71016,  41.63782,  15.02675, -15.02675, -41.63782, -58.71016, -62.33271,
    -51.67558, -29.18021, 0,        0,         30.55692,  57.80252,  78.78433,
    91.22862,  93.78689,  86.18188, 69.23774,  44.79061
  ],
  'z': [
    0,       0,       0.49344, 0.49344, 0.49344, 0.49344, 0.49344, 0.49344, 0.49344,
    0.49344, 1.97327, 1.97327, 1.97327, 1.97327, 1.97327, 1.97327, 1.97327, 1.97327,
    1.97327, 1.97327, 1.97327, 1.97327, 1.97327, 1.97327, 4.43804, 4.43804, 4.43804,
    4.43804, 4.43804, 4.43804, 4.43804, 4.43804, 4.43804
  ],
  'color': [
    4293326232, 4294815329, 4294208835, 4294967231, 4293326232, 4293326232, 4294815329,
    4294208835, 4294967231, 4293326232, 4293326232, 4294815329, 4294208835, 4294967231,
    4293326232, 4293326232, 4294815329, 4294208835, 4294967231, 4293326232, 4293326232,
    4294815329, 4294208835, 4294967231, 4293326232, 4293326232, 4294815329, 4294208835,
    4294967231, 4293326232, 4293326232, 4294815329, 4294208835
  ]
})

const INITIAL_VIEW_STATE = {
  target: [0, 0, 0],
  rotationX: 0,
  rotationOrbit: 0,
  orbitAxis: 'Y',
  fov: 50,
  minZoom: 0,
  maxZoom: 10,
  zoom: 1
};

const transitionInterpolator = new LinearInterpolator(['rotationOrbit']);

export default class App extends PureComponent {
  constructor(props) {
    super(props);

    this.state = {viewState: INITIAL_VIEW_STATE};

    this._onLoad            = this._onLoad.bind(this);
    this._onViewStateChange = this._onViewStateChange.bind(this);
    this._rotateCamera      = this._rotateCamera.bind(this);
  }

  _onViewStateChange({viewState}) { this.setState({viewState}); }

  _rotateCamera() {
    const {viewState} = this.state;
    this.setState({
      viewState: {
        ...viewState,
        rotationOrbit: viewState.rotationOrbit + 120,
        transitionDuration: 2400,
        transitionInterpolator,
        onTransitionEnd: this._rotateCamera
      }
    });
  }

  _onLoad({header, loaderData, progress}) {
    // metadata from LAZ file header
    const {mins, maxs} = loaderData.header;

    if (mins && maxs) {
      // File contains bounding box info
      this.setState({
        viewState: {
          ...this.state.viewState,
          target: [(mins[0] + maxs[0]) / 2, (mins[1] + maxs[1]) / 2, (mins[2] + maxs[2]) / 2],
          /* global window */
          zoom: Math.log2(window.innerWidth / (maxs[0] - mins[0])) - 1
        }
      },
                    this._rotateCamera);
    }

    if (this.props.onLoad) { this.props.onLoad({count: header.vertexCount, progress: 1}); }
  }

  render() {
    const {viewState} = this.state;

    const layers = [new PointCloudLayer({
      id: 'laz-point-cloud-layer',
      // mesh: './public/test3.laz',
      // loaders: [LASLoader],
      // onDataLoad: this._onLoad,
      // getNormal: [0, 1, 0],
      // getColor: [255, 255, 255],
      data: {
        points: {
          pointPositionX: df.get('x').data,
          pointPositionY: df.get('y').data,
          pointPositionZ: df.get('z').data,
          pointColor: df.get('color').data,
        }
      },
      coordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
      opacity: 0.5,
      pointSize: 5
    })];

    return (
      <DeckGL
        views={new OrbitView()}
        viewState={viewState}
        controller={true}
        onViewStateChange={this._onViewStateChange}
        layers={layers}
        parameters={
      { clearColor: [0.93, 0.86, 0.81, 1] }}
      />
    );
  }
}

export function renderToDOM(container) {
  render(<App />, container);
  }
