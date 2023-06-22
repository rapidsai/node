// Copyright (c) 2023, NVIDIA CORPORATION.
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

import './App.css';
import React, { useEffect, useReducer, useState } from 'react';
import Controls from './Controls/Controls';
import ErrorBoundary from './ErrorBoundary';
import Map from './Map';
import Particles from './Particles';
import reducer from './Reducer';
import Title from './Title';
import Spinner from './Spinner';
import { ParticleState, State } from './types'

const initialState: State = {
  zoomLevel: 2.0,
  angle: 0,
  fetching: false,
  pointBudget: 6000000,
  pointsPerRequest: 500000,
  pointOffset: 0,
  quads: {
    displayedPoints: 0,
    pointOffsets: [],
  },
  done: {},
  map: { transform: {} },
  isHeld: false,
  mapReady: false,
  sourceName: 'NAD_r11.txt',
};

function App(): JSX.Element {
  const [state, dispatch] = useReducer(reducer, initialState);
  const [spinning, setSpinning] = useState(true);

  const updateTransformHandler = (event: unknown) => {
    dispatch({ type: 'UPDATE_TRANSFORM', event: event });
  };
  const mapReadyHandler = (event: unknown) => {
    dispatch({ type: 'MAP_READY', event: event });
  };
  const loadingHandler = (event: unknown) => {
    setSpinning(true);
  }
  const updatePointOffsetHandler = (event: unknown) => {
    dispatch({ type: 'UPDATE_POINTOFFSET', event: event });
    setSpinning(false);
  };

  // subscribe event
  useEffect(() => {
    const scrollHandler = (event: WheelEvent) => {
      dispatch({ type: 'SCROLL', event: event });
    };
    const clickHandler = (event: MouseEvent) => {
      dispatch({ type: 'MOUSE_CLICK', event: event });
      console.log(state);
    };
    const releaseHandler = (event: MouseEvent) => {
      dispatch({ type: 'MOUSE_RELEASE', event: event });
    };
    window.addEventListener('wheel', scrollHandler);
    window.addEventListener('mousedown', clickHandler);
    window.addEventListener('mouseup', releaseHandler);

    return () => {
      // unsubscribe event
      window.removeEventListener('wheel', scrollHandler);
      window.removeEventListener('mousedown', clickHandler);
      window.removeEventListener('mouseup', releaseHandler);
    };
  }, [state]);

  return (
    <div className="App">
      <Title />
      <div className="map-box">
        {spinning && <Spinner />}
        <Map updateTransform={updateTransformHandler} mapReady={mapReadyHandler} />
        <ErrorBoundary>
          {state.mapReady ? <Particles {...state} loading={loadingHandler} updatePointOffset={updatePointOffsetHandler} /> : null}
        </ErrorBoundary>
      </div>
      <Controls props={state} />
    </div>
  );
}

export default App;
