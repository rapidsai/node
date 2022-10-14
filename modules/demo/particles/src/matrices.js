/**
 * Copyright (c) 2022 NVIDIA Corporation
 */

const mat4     = require('gl-mat4');
const glmatrix = require('gl-matrix');

const getLookAtZ = (props) => { return 35 * Math.pow(1.2, props.zoomLevel); }

/*
 * LonLat coordinates of the extents of the background image.
 */
export const worldCoords =
  [
    -131.7,
    46.71,  // top right
    -59.2,
    48,  // top left
    -67.8,
    18.9,  // bottom left
    -121.32,
    17.6  // bottom right
  ]

  export const getProjection = (space1, space2) => {
    const w      = space1;
    const s      = space2;
    const wst    = mat4.multiply([], w, mat4.transpose([], s));
    const sst    = mat4.multiply([], s, mat4.transpose([], s));
    const sstInv = mat4.invert([], glmatrix.mat4.add([], sst, mat4.identity([])));
    const A      = mat4.multiply([], wst, sstInv);
    return A;
  };

export const getPointsViewMatrix = (props) => {
  const t = 0;  // 0.015 * (props.angle);

  // s = screen coords
  // w = world coords
  // A = the projection from s to w
  // As = w = Ass^t(ss^t)^-1 = ws^t(ss^t)^-1
  const A = getProjection([...worldCoords, 0, 0, 0, 0, 0, 0, 0, 0], [
    props.screenWidth,
    props.screenHeight,
    0,
    0,
    0,
    props.screenHeight,
    props.screenWidth,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0
  ]);
  const newCenter =
    mat4.multiply([], A, [props.centerX, 0, 0, 0, 0, props.centerY, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
  const result = mat4.lookAt(
    [],
    [-60, 17, 10],
    [-60, 17, 0],
    [0, -1, 0],
  )
  const translation = mat4.translate([], result, [-newCenter[0], -newCenter[5], 0]);
  const rotation    = mat4.rotate([], translation, t, [t, t, 1]);
  return rotation;
};

export const getBackgroundViewMatrix = (props) => {
  const t = 0;  // 0.015 * (props.angle);
  // const result = mat4.lookAt([], [100.5, -38.5, getLookAtZ(props)], [100.5, -38.5, 0], [0, 1,
  // 0]);
  const A = getProjection([...worldCoords, 0, 0, 0, 0, 0, 0, 0, 0], [
    props.screenWidth,
    props.screenHeight,
    0,
    0,
    0,
    props.screenHeight,
    props.screenWidth,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0
  ]);
  const newCenter =
    mat4.multiply([], A, [props.centerX, 0, 0, 0, 0, props.centerY, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
  const result = mat4.lookAt(
    [],
    [133, -12.5, 10],
    [133, -12.5, 0],
    [0, 1, 0],
  );
  const translation = mat4.translate([], result, [-newCenter[0], newCenter[5], 0]);
  const rotation    = mat4.rotate([], translation, t, [t, t, 1]);
  return rotation;
};

export const getCurrentOrthoScale = (props) => { return 35 * Math.pow(1.2, props.zoomLevel);}

export const getPointsProjectionMatrix = (props) => {
  const orthoScale = getCurrentOrthoScale(props);
  return mat4.ortho([], orthoScale, -orthoScale, orthoScale / 6.0, -orthoScale, 1, 1000);
};

export const getBackgroundProjectionMatrix = (props) => {
  const orthoScale = getCurrentOrthoScale(props);
  return mat4.ortho([], -orthoScale, orthoScale, orthoScale / 6.0, -orthoScale, 1, 1000);
};

export const getCurrentWorldBounds = (props) => {
  const A = getProjection(
    [...worldCoords, 0, 0, 0, 0, 0, 0, 0, 0],
    [
      props.screenWidth,
      props.screenHeight,
      0,
      0,
      0,
      props.screenHeight,
      props.screenWidth,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
  );
  const eyeLonLat = mat4.multiply([], A, [
    props.centerX,
    props.centerY,
    0,
    0,
  ])
  const LonLatEye = mat4.multiply([], A, eyeLonLat);
  console.log(LonLatEye);
  const orthoScale = getCurrentOrthoScale(props);
}
