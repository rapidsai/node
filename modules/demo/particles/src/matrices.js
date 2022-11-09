/**
 * Copyright (c) 2022 NVIDIA Corporation
 */

const mat4     = require('gl-mat4');
const glmatrix = require('gl-matrix');

const getLookAtZ = (props) => { return 35 * Math.pow(1.2, props.zoomLevel); }

export const getProjection = (space1, space2) => {
  const w        = space1;
  const s        = space2;
  const wst      = mat4.multiply([], w, mat4.transpose([], s));
  const sst      = mat4.multiply([], s, mat4.transpose([], s));
  const identity = glmatrix.mat4.multiplyScalar([], mat4.identity([]), 0.00001);
  const sstInv   = mat4.invert([], glmatrix.mat4.add([], sst, identity));
  const A        = mat4.multiply([], wst, sstInv);
  return A;
};

export const getPointsWorldProjection =
  (props) => {
    const screenLookAt = [props.centerX, props.centerY, 1, 1];
    const A            = getProjection(props.w(), props.s());
    const lookAtWorld  = mat4.multiply([], A, screenLookAt);
    const lookAtFinal  = mat4.lookAt(
      [], [lookAtWorld[0], lookAtWorld[1], 10], [lookAtWorld[0], lookAtWorld[1], 0], [0, -1, 0]);
    return lookAtFinal;
  }

export const getPointsViewMatrix = (props) => {
  const projection = getPointsWorldProjection(props);
  return projection;
};

export const getBackgroundViewMatrix = (props) => {
  const projection = getPointsWorldProjection(props);
  projection[12]   = -projection[12] - 185;
  projection[13]   = projection[13] - 5;
  return projection;
};

export const getCurrentOrthoScale = (props) => { return 0 + 35 * Math.pow(1.2, props.zoomLevel);}

export const getPointsProjectionMatrix = (props) => {
  const orthoScale = getCurrentOrthoScale(props);
  return mat4.ortho([], orthoScale, -orthoScale, orthoScale / 6.0, -orthoScale, 1, 1000);
};

export const getBackgroundProjectionMatrix = (props) => {
  const orthoScale = getCurrentOrthoScale(props);
  return mat4.ortho([], -orthoScale, orthoScale, orthoScale / 6.0, -orthoScale, 1, 1000);
};

export const getCurrentWorldBounds = (props) => {
  const A          = getProjection(props.w(), props.s());
  const eyeLonLat  = mat4.multiply([], A, [
    props.centerX,
    props.centerY,
    0,
    0,
  ])
  const invert     = mat4.invert([], glmatrix.mat4.add([], A, mat4.identity([])));
  const LonLatEye  = mat4.multiply([], invert, eyeLonLat);
  const orthoScale = getCurrentOrthoScale(props);
  return orthoScale;
}
