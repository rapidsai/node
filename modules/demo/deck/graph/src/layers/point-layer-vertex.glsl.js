// Copyright (c) 2015 - 2017 Uber Technologies, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

export default `\
#define SHADER_NAME node-layer-vertex-shader

precision lowp float;
// precision mediump float;
// precision highp float;

// attribute vec3 positions;

attribute float radius;
attribute float fillColors;
attribute float xPositions;
attribute float yPositions;
attribute float xPositions64Low;
attribute float yPositions64Low;
// attribute vec3 pickingColors;

uniform float opacity;
uniform float radiusScale;
uniform float radiusMinPixels;
uniform float radiusMaxPixels;

varying vec4 vFillColor;

void main(void) {

//   geometry.uv = positions.xy;
//   geometry.pickingColor = pickingColors;

  // set point size
//   float sizeRatio = radius / radiusScale;
  float size = mix(radiusMinPixels, radiusMaxPixels, radius / radiusScale);
// //   float size = project_size_to_pixel(mix(radiusMinPixels, radiusMaxPixels, sizeRatio));
//   gl_PointSize = clamp(size / radiusScale, radiusMinPixels, radiusMaxPixels * sizeRatio);

//   gl_PointSize = clamp(project_size_to_pixel(radius), radiusMinPixels, radiusMaxPixels);

  gl_PointSize = clamp(size, radiusMinPixels, radiusMaxPixels);

  // set point position
  gl_Position = project_position_to_clipspace(
    vec3(xPositions,      yPositions      * 1., 0.), // extruded position
    vec3(xPositions64Low, yPositions64Low * 1., 0.), // extruded position-64-low
    vec3(0., 0., 0.)
  );

  // Dark model
  vFillColor = mix(
    vec4(1, .7, 0, opacity), // female (fillColors != 1.0)
    vec4(.1, .39, .75, opacity), // male   (fillColors == 1.0)
    step(1., fillColors)
  );

  // Light model
  // vFillColor = mix(
  //   vec4(1, 0, .5, opacity), // female (fillColors != 1.0)
  //   vec4(0, .5, 1, opacity), // male   (fillColors == 1.0)
  //   step(1., fillColors)
  // );

  // Set alpha to be inversely-proportional to radius, so as a point gets larger it is more transparent.
  vFillColor.w *= (1. - smoothstep(0., radiusScale, max(radius, 1.)));

  // If gl_PointSize is < 0.1, set the alpha to 0.0 so the point is culled
  vFillColor.w *= step(0.1, gl_PointSize);

//   DECKGL_FILTER_COLOR(vFillColor, geometry);
}
`;
