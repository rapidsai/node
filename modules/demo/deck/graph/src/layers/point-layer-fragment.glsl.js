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
#define SHADER_NAME node-layer-fragment-shader

precision lowp float;
// precision mediump float;
// precision highp float;

varying vec4 vFillColor;

void main(void) {

  // If the alpha channel is already 0, discard
  if (vFillColor.w == 0.0) discard;

  float distanceFromCenter = length(gl_PointCoord * 2. - vec2(1.));

  // If the current pixel is outside the unit circle, discard this
  // pixel. Otherwise if the current pixel is 95-100% units away
  // from the center blend the alpha so the edge appears smooth.
  float edgeAlphaBlend = smoothstep(1.0, 0.95, distanceFromCenter);

  // if current pixel is completely outside the circle, discard it
  if (edgeAlphaBlend == 0.0) discard;

  gl_FragColor = vec4(vFillColor.rgb, vFillColor.a * edgeAlphaBlend);
//   gl_FragColor = picking_filterHighlightColor(gl_FragColor);
//   gl_FragColor = picking_filterPickingColor(gl_FragColor);
}
`;
