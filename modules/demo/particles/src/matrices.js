/**
 * Copyright (c) 2022 NVIDIA Corporation
 */

const regl = require('regl')();
const mat4 = require('gl-mat4');

export const getPointsViewMatrix = (props) => {
  const t           = 0;  // 0.015 * (props.angle);
  const lookAtZ     = 4 * Math.pow(1.2, props.zoomLevel);
  const result      = mat4.lookAt([],
                             [props.centerX / lookAtZ / 10, props.centerY / lookAtZ / 10, lookAtZ],
                             [props.centerX / lookAtZ / 10, props.centerY / lookAtZ / 10, 0],
                             [0, -1, 0]);
  const translation = mat4.translate([], result, [0, 0, 0]);
  const rotation    = mat4.rotate([], translation, t, [t, t, 1]);
  return rotation;
};

export const getBackgroundViewMatrix = (props) => {
  const t       = 0;  // 0.015 * (props.angle);
  const lookAtZ = 4 * Math.pow(1.2, props.zoomLevel);
  const result =
    mat4.lookAt([],
                [-props.centerX / lookAtZ / 10, -props.centerY / lookAtZ / 10, lookAtZ],
                [-props.centerX / lookAtZ / 10, -props.centerY / lookAtZ / 10, 0],
                [0, 1, 0]);
  const translation = mat4.translate([], result, [0, 0, 0]);
  const rotation    = mat4.rotate([], translation, t, [t, t, 1]);
  return rotation;
};

export const getProjectionMatrix = (props) => { return mat4.frustum([], -1, 1, 1, -1, 1, 1000)};
