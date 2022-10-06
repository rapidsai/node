/**
 * Copyright (c) Mik BRY
 * mik@miklabs.com
 *
 * This source code is licensed under private license found in the
 * LICENSE file in the root directory of this source tree.
 */

import React, { useEffect } from 'react';
import PropTypes from 'prop-types';
import './App.css';
import ParticlesCanvas from './ParticlesCanvas';
import ParticlesView from './ParticlesView';
import BackgroundView from './BackgroundView';

const initialState = {
  angle: 0.0,
  mouseX: 0,
  mouseY: 0,
  isHeld: false,
  zoomLevel: 1.0
}

const toggle = previous => !previous;
const reducer = (state, action) => {
  switch (action.type) {
    case 'MOUSE_MOVE':
      return {
        ...state,
        mouseX: action.event.screenX,
        mouseY: action.event.screenY,
      }
    case 'MOUSE_CLICK':
      console.log('click');
      return {
        ...state,
        isHeld: toggle()
      }
    case 'MOUSE_RELEASE':
      console.log('unclick');
      return {
        ...state,
        isHeld: toggle()
      }
    case 'SCROLL':
      const zoom = action.event.deltaY > 0 ? 1 : -1;
      const result = state.zoomLevel + zoom;
      return {
        ...state,
        zoomLevel: result
      }
    default:
      throw new Error('chalupa batman');
  }
}

function App() {
  const [state, dispatch] = React.useReducer(reducer, initialState);

  const overHandler = (event) => {
    //dispatch({ type: 'MOUSE_MOVE', event: event });
  }
  const scrollHandler = (event) => {
    dispatch({ type: 'SCROLL', event: event });
  }
  const clickHandler = (event) => {
    dispatch({ type: 'MOUSE_CLICK', event: event });
  }
  const releaseHandler = (event) => {
    dispatch({ type: 'MOUSE_RELEASE', event: event });
  }
  useEffect(() => {
    // subscribe event
    window.addEventListener("wheel", scrollHandler);
    window.addEventListener("mousemove", overHandler);
    window.addEventListener("mousedown", clickHandler);
    window.addEventListener("mouseup", releaseHandler);
    return () => {
      // unsubscribe event
      window.removeEventListener("wheel", scrollHandler);
      window.removeEventListener("mousemove", overHandler);
      window.removeEventListener("mousedown", clickHandler);
      window.removeEventListener("mouseup", releaseHandler);
    };
  }, []);

  return (
    <div className="App">
      <div className="BackgroundView">
        <BackgroundView appState={state.zoomLevel} />
      </div>
      <div className="ParticlesCanvas" >
        <ParticlesCanvas />
      </div>
      <div className="ParticlesView" >
        <ParticlesView zoomLevel={state.zoomLevel} angle={state.angle} />
      </div>
      <div className="App-title">WebGL React App</div>
    </div >
  );
}

export default App;
