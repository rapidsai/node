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
import ParticlesView from './ParticlesView';
import BackgroundView from './BackgroundView';

function App({ scene }) {
  const overHandler = (event) => {
    console.log('mouse over');
    console.log(event);
  }
  const scrollHandler = (event) => {
    console.log('scroll');
    console.log(event);
  }
  const clickHandler = (event) => {
    console.log('click');
    console.log(event);
  }
  useEffect(() => {
    // subscribe event
    window.addEventListener("wheel", scrollHandler);
    window.addEventListener("mousemove", overHandler);
    window.addEventListener("mousedown", clickHandler);
    return () => {
      // unsubscribe event
      window.removeEventListener("wheel", scrollHandler);
      window.removeEventListener("mousemove", overHandler);
      window.removeEventListener("mousedown", clickHandler);
    };
  }, []);

  return (
    <div className="App">
      <div className="BackgroundView">
        <BackgroundView />
      </div>
      <div className="ParticlesView" >
        <ParticlesView />
      </div>
      <div className="App-title">WebGL React App</div>
    </div >
  );
}

App.propTypes = {
  // eslint-disable-next-line react/forbid-prop-types
  scene: PropTypes.object.isRequired,
};

export default App;
