// Copyright (c) 2021, NVIDIA CORPORATION.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

const goldenRatioConjugate = 0.618033988749895;

export class ColorMapper {
  constructor(hue = 0.99, saturation = 0.99, brightness = 0.99) {
    this._h = hue % 1;
    this._s = saturation % 1;
    this._v = brightness % 1;
    this._map = Object.create(null);
  }
  get(id, colors = [], index = 0) {
    [
      colors[index * 4 + 0],
      colors[index * 4 + 1],
      colors[index * 4 + 2]
    ] = this._map[id] || (this._map[id] = this.generate());
    colors[index * 4 + 3] = 255;
    return colors;
  }
  generate() {
    const rgb = HSVtoRGB(this._h, this._s, this._v);
    this._h = (this._h + goldenRatioConjugate) % 1;
    return rgb;
  }
}

// # HSV values in [0..1]
// # returns [r, g, b] values from 0 to 255
function HSVtoRGB(h, s, v) {
  var r, g, b, i, f, p, q, t;
  if (arguments.length === 1) {
    s = h.s, v = h.v, h = h.h;
  }
  i = Math.floor(h * 6);
  f = h * 6 - i;
  p = v * (1 - s);
  q = v * (1 - f * s);
  t = v * (1 - (1 - f) * s);
  switch (i % 6) {
    case 0: r = v, g = t, b = p; break;
    case 1: r = q, g = v, b = p; break;
    case 2: r = p, g = v, b = t; break;
    case 3: r = p, g = q, b = v; break;
    case 4: r = t, g = p, b = v; break;
    case 5: r = v, g = p, b = q; break;
  }
  return [
    Math.round(r * 255), // r
    Math.round(g * 255), // g
    Math.round(b * 255), // b
  ]
}
