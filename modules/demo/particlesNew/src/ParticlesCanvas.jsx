/**
 * Copyright (c) Mik BRY
 * mik@mikbry.com
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

import React, { useEffect, useRef } from 'react';

const ParticlesCanvas = (props) => {
  const ref = useRef();
  return <canvas ref={ref} id="reglCanvas" />;
}

export default ParticlesCanvas;
