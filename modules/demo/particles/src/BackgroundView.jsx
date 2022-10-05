/**
 * Copyright (c) Mik BRY
 * mik@mikbry.com
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

import React, { useEffect, useRef } from 'react';
import usa_map from "../public/images/usa_map.png";

const BackgroundView = () => {
  return (
    <div className="background" style={{
      backgroundImage: `url(${usa_map})`,
      backgroundSize: "contain",
      backgroundRepeat: "no-repeat",
      height: "100%",
      width: "100%",
    }}>
      Hello World
      <div className="foreground"></div>
    </div>
  )
};

export default BackgroundView;
