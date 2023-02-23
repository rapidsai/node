import './App.css';

import React, {useEffect} from 'react';
import Controls from './Controls.js';
import Map from './Map.js';
import Particles from './Particles.js';
import reducer from './Reducer';
import Title from './Title.js';

const initialState = {
  angle: 0.0,
  mouseX: 0,
  mouseY: 0,
  centerX: 0,
  centerY: 0,
  isHeld: false,
  zoomLevel: 1.0
}

function
App() {
  const [state, dispatch] = React.useReducer(reducer, initialState);

  useEffect(() => {
    // subscribe event
    const overHandler = (event) => { dispatch({type: 'MOUSE_MOVE', event: event}); };
    const scrollHandler = (event) => { dispatch({type: 'SCROLL', event: event}); };
    const clickHandler = (event) => { dispatch({type: 'MOUSE_CLICK', event: event}); };
    const releaseHandler = (event) => { dispatch({type: 'MOUSE_RELEASE', event: event}); };

    window.addEventListener('wheel', scrollHandler);
    window.addEventListener('mousemove', overHandler);
    window.addEventListener('mousedown', clickHandler);
    window.addEventListener('mouseup', releaseHandler);

    return () => {
      // unsubscribe event
      window.removeEventListener('wheel', scrollHandler);
      window.removeEventListener('mousemove', overHandler);
      window.removeEventListener('mousedown', clickHandler);
      window.removeEventListener('mouseup', releaseHandler);
    };
  }, []);

  return (
    <div className='App'>
      <Title/>
      <div className='map-box'>
        <Map/>
        <Particles/>
      </div>
      <Controls/>
    </div>
  );
}

export default App;
