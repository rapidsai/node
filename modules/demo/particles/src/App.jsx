/**
 * Copyright (c) Mik BRY
 * mik@miklabs.com
 *
 * This source code is licensed under private license found in the
 * LICENSE file in the root directory of this source tree.
 */

import React from 'react';
import PropTypes from 'prop-types';
import './App.css';
import GLView from './GLView';
import ParticlesView from './ParticlesView';
import BackgroundView from './BackgroundView';

function App({ scene }) {
  return (
    <div className="App">
      <div className="BackgroundView" >
        <BackgroundView />
      </div>
      <div className="ParticlesView" >
        <ParticlesView />
      </div>
      <div className="App-title">WebGL React App</div>
    </div>
  );
}

App.propTypes = {
  // eslint-disable-next-line react/forbid-prop-types
  scene: PropTypes.object.isRequired,
};

export default App;
