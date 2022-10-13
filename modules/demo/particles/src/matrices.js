/**
 * Copyright (c) 2022 NVIDIA Corporation
 */

const regl = require('regl')();
const mat4 = require('gl-mat4');

const getLookAtZ = (props) => { return 184 * Math.pow(1.2, props.zoomLevel); }

export const getPointsViewMatrix = (props) => {
  const t = 0;  // 0.015 * (props.angle);
  /*
  const result      = mat4.lookAt([],
                             [props.centerX / lookAtZ / 10, props.centerY / lookAtZ / 10, lookAtZ],
                             [props.centerX / lookAtZ / 10, props.centerY / lookAtZ / 10, 0],
                             [0, -1, 0]);
  */
  const result = mat4.lookAt(
    [],
    [-112.5, 45.5, getLookAtZ(props)],
    [-112.5, 45.5, 0],
    [0, -1, 0],
  )
  const translation = mat4.translate([], result, [0, 0, 0]);
  const rotation    = mat4.rotate([], translation, t, [t, t, 1]);
  return rotation;
};

export const getBackgroundViewMatrix = (props) => {
  const t      = 0;  // 0.015 * (props.angle);
  const result = mat4.lookAt([], [112.5, -45.5, getLookAtZ(props)], [112.5, -45.5, 0], [0, 1, 0]);
  /*
const result = mat4.lookAt(
[],
[112.5, -45.5, lookAtZ],
[112.5, -45.5, 0],
[0, -1, 0],
)
*/
  const translation = mat4.translate([], result, [0, 0, 0]);
  const rotation    = mat4.rotate([], translation, t, [t, t, 1]);
  return rotation;
};

export const getProjectionMatrix = (props) => { return mat4.frustum([], -1, 1, 1, -1, 1, 1000)};
