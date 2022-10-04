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
    <img src={usa_map} alt="The background" />
  )
};

export default BackgroundView;
