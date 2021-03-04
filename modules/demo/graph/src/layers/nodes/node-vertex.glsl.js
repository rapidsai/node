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
#version 300 es
#define SHADER_NAME node-vertex-shader

precision highp float;

uniform bool filled;
uniform float stroked;
uniform float fillOpacity;
uniform float strokeRatio;
uniform float strokeOpacity;
uniform float radiusScale;
uniform float radiusMinPixels;
uniform float radiusMaxPixels;
uniform float lineWidthScale;
uniform float lineWidthMinPixels;
uniform float lineWidthMaxPixels;
uniform uint highlightedNode;
uniform uint highlightedSourceNode;
uniform uint highlightedTargetNode;

in vec3 positions;
in float instanceRadius;
in uint instanceNodeIndices;
// in float instanceLineWidths;
in vec4 instanceFillColors;
in vec4 instanceLineColors;
in float instanceXPositions;
in float instanceYPositions;
in float instanceXPositions64Low;
in float instanceYPositions64Low;
in vec3 instancePickingColors;

out vec4 vFillColor;
out vec4 vLineColor;
out vec2 unitPosition;
out float innerUnitRadius;
out float outerRadiusPixels;

void main(void) {
    geometry.worldPosition = vec3(instanceXPositions, instanceYPositions, 0.);

    // Multiply out radius and clamp to limits
    outerRadiusPixels = project_size_to_pixel(instanceRadius * radiusScale);
    outerRadiusPixels = clamp(outerRadiusPixels, radiusMinPixels, radiusMaxPixels);

    // Multiply out line width and clamp to limits
    float lineWidthPixels = 0.;
    lineWidthPixels = outerRadiusPixels * strokeRatio * lineWidthScale;
    lineWidthPixels = clamp(lineWidthPixels, lineWidthMinPixels, lineWidthMaxPixels);

    // outer radius needs to offset by half stroke width
    outerRadiusPixels += stroked * lineWidthPixels / 2.0;

    // position on the containing square in [-1, 1] space
    unitPosition = positions.xy;
    geometry.uv = unitPosition;
    geometry.pickingColor = instancePickingColors;

    innerUnitRadius = 1.0 - stroked * lineWidthPixels / outerRadiusPixels;
  
    vec3 offset = positions * project_pixel_size(outerRadiusPixels);
    DECKGL_FILTER_SIZE(offset, geometry);
    gl_Position = project_position_to_clipspace(
        vec3(instanceXPositions, instanceYPositions, 0.),
        vec3(instanceXPositions64Low, instanceYPositions64Low, 0.),
        offset, geometry.position
    );
    DECKGL_FILTER_GL_POSITION(gl_Position, geometry);

    // Apply opacity to instance color, or return instance picking color
    vFillColor = vec4(instanceFillColors.rgb, fillOpacity);
    DECKGL_FILTER_COLOR(vFillColor, geometry);
    vLineColor = vec4(instanceLineColors.rgb, strokeOpacity);
    DECKGL_FILTER_COLOR(vLineColor, geometry);

    picking_vRGBcolor_Avalid.a = float(
        bool(picking_vRGBcolor_Avalid.a) ||
        instanceNodeIndices == highlightedNode ||
        instanceNodeIndices == highlightedSourceNode ||
        instanceNodeIndices == highlightedTargetNode );
}
`;
