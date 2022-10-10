/**
 * Copyright (c) Mik BRY
 * mik@mikbry.com
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

import React, { useEffect } from 'react';
import usa_map from "./usa_map.png";

const BackgroundView = (props) => {
  useEffect(() => {
    console.log(props.zoomLevel);
  });
  return (
    <div id="background" style={{
      backgroundImage: `url(${usa_map})`,
      backgroundSize: 'cover',
      height: 40 * (50 - (25 + props.zoomLevel)),
      width: 40 * (50 - (25 + props.zoomLevel)),
      backgroundRepeat: "no-repeat",
    }}>
    </div>
  )
};

export default BackgroundView;
