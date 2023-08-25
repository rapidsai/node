// Copyright (c) 2020, NVIDIA CORPORATION.
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

import {EventEmitter} from 'events';

import {glfw, GLFWmonitor} from './glfw';

export class Monitor extends EventEmitter {
  constructor(id: GLFWmonitor, connected = true) {
    super();
    this._id        = id;
    this._connected = connected;
    this._name      = glfw.getMonitorName(id);
    if (connected) {
      ({x: this._x, y: this._y, width: this._width, height: this._height} =
         glfw.getMonitorWorkarea(id));
      ({xscale: this._xscale, yscale: this._yscale} = glfw.getMonitorContentScale(id));
      ({width: this._widthMM, height: this._heightMM} = glfw.getMonitorPhysicalSize(id));
    }
  }

  private _x = 0;
  public get x() { return this._x; }

  private _y = 0;
  public get y() { return this._y; }

  private _id: GLFWmonitor = 0;
  public get id() { return this._id; }

  private _name = '';
  public get name() { return this._name; }

  private _xscale = 0;
  public get xscale() { return this._xscale; }

  private _yscale = 0;
  public get yscale() { return this._yscale; }

  private _width = 800;
  public get width() { return this._width; }

  private _height = 600;
  public get height() { return this._height; }

  private _widthMM = 800;
  public get widthMM() { return this._widthMM; }

  private _heightMM = 600;
  public get heightMM() { return this._heightMM; }

  private _connected = true;
  public get connected() { return this._connected; }

  public[Symbol.toStringTag]: string;
  public inspect() { return this.toString(); }
  public[Symbol.for('nodejs.util.inspect.custom')]() { return this.toString(); }
  public toString() {
    return `${this[Symbol.toStringTag]} ${JSON.stringify({
      id: this.id,
      x: this.x,
      y: this.y,
      width: this.width,
      height: this.height,
      xscale: this.xscale,
      yscale: this.yscale,
      widthMM: this.widthMM,
      heightMM: this.heightMM,
    })}`;
  }
}

Monitor.prototype[Symbol.toStringTag] = 'Monitor';
