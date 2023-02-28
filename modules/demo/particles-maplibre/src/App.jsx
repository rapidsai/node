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

import React, { useEffect, useState } from 'react';

import Controls from './Controls/Controls.jsx';
import ErrorBoundary from './ErrorBoundary.jsx';
import Map from './Map.js';
import Particles from './Particles.jsx';
import reducer from './Reducer';
import Title from './Title.js';

const initialState = {
  /*
   All application state except for a few constants
   are defined here. This state must be shared between
   components.
   */
  zoomLevel: 2.0,
  angle: 0,
  fetching: false,
  pointBudget: 250000000,
  pointsPerRequest: 1000000,
  pointOffset: 0,
  quads: {
    totalPoints: undefined,
    displayedPoints: 0,
    pointOffsets: [],
  },
  done: {},
  map: {
    transform: {
      mercatorMatrix: [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0
      ],
    }
  },
};

function App() {
  const [state, dispatch] = React.useReducer(reducer, initialState);

  const updateTransformHandler = (event) => {
    dispatch({ type: 'UPDATE_TRANSFORM', event: event });
  }
  const mapReadyHandler = (event) => {
    dispatch({
      type: 'MAP_READY', event: event
    });
  }

  // subscribe event
  useEffect(() => {
    const scrollHandler = (event) => {
      dispatch({ type: 'SCROLL', event: event });
    }
    const clickHandler = (event) => {
      dispatch({ type: 'MOUSE_CLICK', event: event });
      console.log(state);
    }
    const releaseHandler = (event) => {
      dispatch({ type: 'MOUSE_RELEASE', event: event });
    }
    window.addEventListener("wheel", scrollHandler);
    window.addEventListener("mousedown", clickHandler);
    window.addEventListener("mouseup", releaseHandler);

    return () => {
      // unsubscribe event
      window.removeEventListener("wheel", scrollHandler);
      window.removeEventListener("mousedown", clickHandler);
      window.removeEventListener("mouseup", releaseHandler);
    };
  }, []);

  return (
    <div className='App'>
      <Title />
      <div className='map-box'>
        <Map props={state} updateTransform={updateTransformHandler} mapReady={mapReadyHandler} />
        <ErrorBoundary>
          {state.mapReady ? <Particles props={state} /> : null}
        </ErrorBoundary>
      </div>
      <Controls props={state} />
    </div>
  );
}

export default App;
