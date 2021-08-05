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

import {
  glfw,
  GLFWClientAPI,
  GLFWContextCreationAPI,
  GLFWInputMode,
  GLFWModifierKey,
  GLFWOpenGLProfile,
  GLFWWindowAttribute,
} from '@nvidia/glfw';
import * as jsdom from 'jsdom';
import {DOMWindow} from 'jsdom';
import {Subscription} from 'rxjs';

import {
  isAltKey,
  isCapsLock,
  isCtrlKey,
  isMetaKey,
  isShiftKey
} from '../../../glfw/src/events/event';

import GLFW, {GLFWStandardCursor} from '../../../glfw/src/glfw';
import {GLFWDOMWindow} from '../../../glfw/src/jsdom/window';
import {dndEvents, GLFWDndEvent} from './events/dnd';
import {GLFWKeyboardEvent, keyboardEvents} from './events/keyboard';
import {GLFWMouseEvent, mouseEvents} from './events/mouse';
import {GLFWWheelEvent, wheelEvents} from './events/wheel';
import {GLFWWindowEvent, windowEvents} from './events/window';
// import {dndEvents, GLFWDndEvent} from './events/dnd';
// import {GLFWKeyboardEvent, keyboardEvents} from './events/keyboard';
// import {GLFWMouseEvent, mouseEvents} from './events/mouse';
// import {GLFWWheelEvent, wheelEvents} from './events/wheel';
// import {GLFWWindowEvent, windowEvents} from './events/window';

// export type GLFWDOMWindowOptions = {
//   x?: number;
//   y?: number;
//   debug?: boolean;
//   width?: number;
//   height?: number;
//   visible?: boolean;
//   decorated?: boolean;
//   resizable?: boolean;
//   transparent?: boolean;
//   devicePixelRatio?: number;
//   openGLMajorVersion?: number;
//   openGLMinorVersion?: number;
//   openGLForwardCompat?: boolean;
//   openGLProfile?: GLFWOpenGLProfile;
//   openGLClientAPI?: GLFWClientAPI;
//   openGLContextCreationAPI?: GLFWContextCreationAPI;
// };

// let rootWindow: GLFWDOMWindow|undefined = undefined;

// export class GLFWDOMWindow {
//   constructor(options: GLFWDOMWindowOptions = {}) { this.init(options); }

//   public init(options: GLFWDOMWindowOptions = {}) {
//     Object.assign(this,
//                   {
//                     debug: false,
//                     openGLMajorVersion: 4,
//                     openGLMinorVersion: 6,
//                     openGLForwardCompat: true,
//                     openGLProfile: GLFWOpenGLProfile.ANY,
//                     openGLClientAPI: GLFWClientAPI.OPENGL,
//                     openGLContextCreationAPI: GLFWContextCreationAPI.EGL,
//                     _title: 'Untitled',
//                     _x: 0,
//                     _y: 0,
//                     _width: 800,
//                     _height: 600,
//                     _mouseX: 0,
//                     _mouseY: 0,
//                     _scrollX: 0,
//                     _scrollY: 0,
//                     _buttons: 0,
//                     _modifiers: 0,
//                     _xscale: 1,
//                     _yscale: 1,
//                     _devicePixelRatio: 1,
//                     _focused: false,
//                     _minimized: false,
//                     _maximized: false,
//                     _swapInterval: 0,
//                     _visible: true,
//                     _decorated: true,
//                     _transparent: false,
//                     _resizable: true,
//                     _forceNewWindow: false,
//                     _subscriptions: new Subscription()
//                   },
//                   options);

//     const validatePropType = (name: keyof this, type: string) => {
//       // eslint-disable-next-line @typescript-eslint/restrict-template-expressions
//       if (typeof this[name] !== type) { throw new TypeError(`options.${name} must be a ${type}`);
//       }
//     };

//     ([
//       'x',
//       'y',
//       'width',
//       'height',
//       'mouseX',
//       'mouseY',
//       'scrollX',
//       'scrollY',
//       'buttons',
//       'modifiers',
//       'xscale',
//       'yscale',
//       'devicePixelRatio',
//       'swapInterval',
//     ] as (keyof this)[])
//       .forEach((prop) => validatePropType(prop, 'number'));

//     ([
//       'debug',
//       'focused',
//       'minimized',
//       'maximized',
//       'visible',
//       'decorated',
//       'transparent',
//       'resizable',
//     ] as (keyof this)[])
//       .forEach((prop) => validatePropType(prop, 'boolean'));

//     this._frameBufferWidth  = this._width;
//     this._frameBufferHeight = this._height;

//     return this._create();
//   }

//   protected _x = 0;
//   public get x() { return this._x; }
//   public set x(_: number) { this._x = _; }

//   protected _y = 0;
//   public get y() { return this._y; }
//   public set y(_: number) { this._y = _; }

//   protected _id: GLFWwindow = 0;
//   public get id() { return this._id; }

//   protected _title = '';
//   public get title() { return this._title; }

//   protected _event: any = undefined;
//   public get event() { return this._event; }

//   protected _cursor = GLFWStandardCursor.ARROW;
//   public get cursor() { return this._cursor; }
//   public set cursor(_: any) {
//     switch (_) {
//       case 'pointer': _ = GLFWStandardCursor.HAND; break;
//       case 'text': _ = GLFWStandardCursor.IBEAM; break;
//       case 'crosshair': _ = GLFWStandardCursor.CROSSHAIR; break;
//       case 'e-resize':
//       case 'w-resize':
//       case 'ew-resize': _ = GLFWStandardCursor.HRESIZE; break;
//       case 'n-resize':
//       case 's-resize':
//       case 'ns-resize': _ = GLFWStandardCursor.VRESIZE; break;
//       case 'auto':
//       case 'none':
//       case 'grab':
//       case 'grabbing':
//       case 'default':
//       default: _ = GLFWStandardCursor.ARROW; break;
//     }
//     if (this._cursor !== _) {
//       this._cursor = _;
//       if (this._id > 0) {
//         switch (_) {
//           case GLFWStandardCursor.ARROW:
//           case GLFWStandardCursor.IBEAM:
//           case GLFWStandardCursor.CROSSHAIR:
//           case GLFWStandardCursor.HAND:
//           case GLFWStandardCursor.HRESIZE:
//           case GLFWStandardCursor.VRESIZE: glfw.setCursor(this._id, _); break;
//           default: break;
//         }
//       }
//     }
//   }

//   protected _width = 800;
//   public get width() { return this._width; }
//   public set width(_: number) { this._width = this._cssToNumber('width', _); }

//   protected _height = 600;
//   public get height() { return this._height; }
//   public set height(_: number) { this._height = this._cssToNumber('height', _); }

//   protected _xscale = 1;
//   public get xscale() { return this._xscale; }

//   protected _yscale = 1;
//   public get yscale() { return this._yscale; }

//   protected  _mouseX = 0;
//   public get mouseX() { return this._mouseX; }

//   protected _mouseY = 0;
//   public get mouseY() { return this._mouseY; }

//   protected _scrollX = 0;
//   public get scrollX() { return this._scrollX; }
//   public set scrollX(_: number) { this._scrollX = _; }

//   protected _scrollY = 0;
//   public get scrollY() { return this._scrollY; }
//   public set scrollY(_: number) { this._scrollY = _; }

//   protected _buttons = 0;
//   public get buttons() { return this._buttons; }

//   protected _focused = false;
//   public get focused() { return this._focused; }

//   protected _minimized = false;
//   public get minimized() { return this._minimized; }

//   protected _maximized = false;
//   public get maximized() { return this._maximized; }

//   protected _swapInterval = 0;
//   public get swapInterval() { return this._swapInterval; }
//   public set swapInterval(_: number) {
//     if (this._swapInterval !== _) {
//       if (this._id > 0 && typeof _ === 'number') { glfw.swapInterval(this._swapInterval = _); }
//     }
//   }

//   protected _modifiers = 0;
//   public get modifiers() { return this._modifiers; }

//   public get altKey() { return isAltKey(this._modifiers); }
//   public get ctrlKey() { return isCtrlKey(this._modifiers); }
//   public get metaKey() { return isMetaKey(this._modifiers); }
//   public get shiftKey() { return isShiftKey(this._modifiers); }
//   public get capsLock() { return isCapsLock(this._modifiers); }

//   protected _devicePixelRatio = 1;
//   public get devicePixelRatio() { return this._devicePixelRatio; }
//   public set devicePixelRatio(_: number) { this._devicePixelRatio = _; }

//   public readonly debug!: boolean;
//   public readonly openGLMajorVersion!: number;
//   public readonly openGLMinorVersion!: number;
//   public readonly openGLForwardCompat!: boolean;
//   public readonly openGLProfile!: GLFWOpenGLProfile;
//   public readonly openGLClientAPI!: GLFWClientAPI;
//   public readonly openGLContextCreationAPI!: GLFWContextCreationAPI;

//   protected _visible = true;
//   public get visible() { return this._visible; }
//   public set visible(_: boolean) {
//     if (this._visible !== _) { ((this._visible = _)) ? this.show() : this.hide(); }
//   }

//   protected _decorated = true;
//   public get decorated() { return this._decorated; }
//   public set decorated(_: boolean) {
//     if (this._decorated !== _) {
//       this._decorated = _;
//       if (this._id > 0) { glfw.setWindowAttrib(this._id, GLFWWindowAttribute.DECORATED, _); }
//     }
//   }

//   protected _transparent = false;
//   public get transparent() { return this._transparent; }
//   public set transparent(_: boolean) {
//     if (this._transparent !== _) { this._transparent = _; }
//   }

//   protected _resizable = true;
//   public get resizable() { return this._resizable; }
//   public set resizable(_: boolean) {
//     if (this._resizable !== _) {
//       this._resizable = _;
//       if (this._id > 0) { glfw.setWindowAttrib(this._id, GLFWWindowAttribute.RESIZABLE, _); }
//     }
//   }

//   public get style() {
//     // eslint-disable-next-line @typescript-eslint/no-this-alias
//     const self = this;
//     return {
//       get width() { return self.width; },
//       set width(_: number) { self.width = _; },
//       get height() { return self.height; },
//       set height(_: number) { self.height = _; },
//       get cursor() { return self.cursor; },
//       set cursor(_: any) { self.cursor = _; },
//     };
//   }

//   public setAttribute(name: any, value: any) {
//     if (name in this) { (this as any)[name] = value; }
//   }

//   public getBoundingClientRect() {
//     return {
//       x: 0,
//       y: 0,
//       width: this.width,
//       height: this.height,
//       left: 0,
//       top: 0,
//       right: this.width,
//       bottom: this.height,
//     };
//   }

//   protected _frameBufferWidth = this._width;
//   public get frameBufferWidth() { return this._frameBufferWidth; }

//   protected _frameBufferHeight = this._height;
//   public get frameBufferHeight() { return this._frameBufferHeight; }

//   protected _forceNewWindow = false;
//   // @ts-ignore
//   protected _subscriptions: Subscription;
//   protected _monitor: Monitor|undefined;

//   public destroyGLFWWindow() { this._destroyGLFWWindow(); }
//   public show() {
//     this.visible = true;
//     this._id || this._create();
//     glfw.showWindow(this._id);
//     return this;
//   }
//   public hide() {
//     this.visible = false;
//     this._id || this._create();
//     glfw.hideWindow(this._id);
//     return this;
//   }
//   public reparent(parent: GLFWParentWindow, x = 0, y = 0) {
//     if (this._id) { glfw.reparentWindow(this._id, parent, x, y); }
//     return this;
//   }

//   protected _create() {
//     if (this._id) { return; }
//     try {
//       let root = null;
//       if (!this._forceNewWindow && rootWindow) { root = rootWindow.id; }
//       const monitor = this._monitor ? this._monitor.id : null;

//       glfw.windowHint(GLFWWindowAttribute.SAMPLES, 4);
//       glfw.windowHint(GLFWWindowAttribute.DOUBLEBUFFER, true);
//       glfw.windowHint(GLFWWindowAttribute.FOCUSED, this.focused);
//       glfw.windowHint(GLFWWindowAttribute.FOCUS_ON_SHOW, this.focused);
//       glfw.windowHint(GLFWWindowAttribute.VISIBLE, this.visible);
//       glfw.windowHint(GLFWWindowAttribute.DECORATED, this.decorated);
//       glfw.windowHint(GLFWWindowAttribute.RESIZABLE, this.resizable);
//       glfw.windowHint(GLFWWindowAttribute.TRANSPARENT_FRAMEBUFFER, this.transparent);

//       glfw.windowHint(GLFWWindowAttribute.CLIENT_API, this.openGLClientAPI);
//       glfw.windowHint(GLFWWindowAttribute.OPENGL_DEBUG_CONTEXT, this.debug);
//       glfw.windowHint(GLFWWindowAttribute.OPENGL_PROFILE, this.openGLProfile);
//       glfw.windowHint(GLFWWindowAttribute.CONTEXT_VERSION_MAJOR, this.openGLMajorVersion);
//       glfw.windowHint(GLFWWindowAttribute.CONTEXT_VERSION_MINOR, this.openGLMinorVersion);
//       glfw.windowHint(GLFWWindowAttribute.OPENGL_FORWARD_COMPAT, this.openGLForwardCompat);
//       glfw.windowHint(GLFWWindowAttribute.CONTEXT_CREATION_API, this.openGLContextCreationAPI);

//       const id = glfw.createWindow(this.width, this.height, this.title, monitor, root);

//       this._id = id;

//       // glfw.setInputMode(this._id, GLFWInputMode.LOCK_KEY_MODS, true);
//       glfw.setInputMode(id, GLFWInputMode.CURSOR, GLFW.CURSOR_NORMAL);
//       glfw.makeContextCurrent(id);

//       ({x: this._x, y: this._y} = glfw.getWindowPos(id));
//       ({width: this._width, height: this._height} = glfw.getWindowSize(id));
//       ({xscale: this._xscale, yscale: this._yscale} = glfw.getWindowContentScale(id));

//       !this._forceNewWindow && !rootWindow && (rootWindow = this);
//       this._frameBufferWidth  = this._width * this._xscale;
//       this._frameBufferHeight = this._height * this._yscale;
//       this._subscriptions && this._subscriptions.unsubscribe();
//       this._subscriptions = new Subscription();
//       this.cursor         = this._cursor;

//       glfw.swapInterval(this.swapInterval);
//       glfw.swapBuffers(id);

//       // [dndEvents(this).subscribe(onGLFWDndEvent.bind(this)),
//       //  mouseEvents(this).subscribe(onGLFWMouseEvent.bind(this)),
//       //  wheelEvents(this).subscribe(onGLFWWheelEvent.bind(this)),
//       //  windowEvents(this).subscribe(onGLFWWindowEvent.bind(this)),
//       //  keyboardEvents(this).subscribe(onGLFWKeyboardEvent.bind(this)),
//       // ].forEach((subscription) => this._subscriptions.add(subscription));
//     } catch (e) {
//       console.error('Error creating window:', e);
//       this._destroyGLFWWindow();
//       throw e;
//     }
//   }

//   protected _destroyGLFWWindow() {
//     const id = this._id;
//     this._subscriptions.unsubscribe();
//     if (id) {
//       this._id = <any>undefined;
//       glfw.destroyWindow(id);
//       if (!this._forceNewWindow && rootWindow === this) { setImmediate(() => process.exit(0)); }
//     }
//   }

//   public[Symbol.toStringTag]: string;
//   public inspect() { return this.toString(); }
//   public[Symbol.for('nodejs.util.inspect.custom')]() { return this.toString(); }
//   public toString() {
//     return `${this[Symbol.toStringTag]} {
//             id: ${this.id},
//             x: ${this.x},
//             y: ${this.y},
//             width: ${this.width},
//             height: ${this.height}
//         }`;
//   }

//   protected _gl: any = null;
//   protected get _clearMask() { return this._gl ? this._gl._clearMask : 0; }
//   protected set _clearMask(_: any) { this._gl && (this._gl._clearMask = _); }

//   protected _cssToNumber(prop: keyof this, value: any): number {
//     switch (typeof value) {
//       case 'number': return value;
//       case 'string': {
//         if (value.endsWith('px')) {
//           return +(value.slice(0, -1));
//         } else if (value.endsWith('%')) {
//           return +this[prop] * (+value.slice(0, -1) / 100);
//         }
//       }
//     }
//     return ((value = +value) !== value) ? +this[prop] : value;
//   }
// }

export function installGLFWWindow(window: jsdom.DOMWindow) {
  const rootWindow: GLFWDOMWindow|undefined = undefined;

  // Attatching properties

  Object.defineProperties(window, {
    id: {get() { return this._id; }, set(_) { this._id = _; }},
    width: {
      value: 800,
      writable: true,
    },
    height: {
      value: 600,
      writable: true,
    },
    title: {
      value: 'Untitled',
      writable: true,
    },
    x: {
      value: 0,
      writable: true,
    },
    y: {
      value: 0,
      writable: true,
    },
    xscale: {
      value: 1,
    },
    yscale: {
      value: 1,
    },
    mouseX: {
      value: 0,
    },
    mouseY: {
      value: 0,
    },
    scrollX: {
      value: 0,
      writable: true,
    },
    scrollY: {
      value: 0,
      writable: true,
    },
    buttons: {
      value: 0,
    },
    focused: {
      value: false,
    },
    minimized: {
      value: false,
    },
    maximized: {
      value: false,
    },
    devicePixelRatio: {
      value: 1,
      writable: true,
    },
    event: {
      value: undefined,
    },

    swapInterval: {
      get() { return this._swapInterval; },
      set(_: number) {
        if (this.swapInterval !== _) {
          if (this._id > 0 && typeof _ === 'number') { glfw.swapInterval(this._swapInterval = _); }
        }
      },
    },

    modifiers: {
      value: 0,
    },

    altKey: {get: () => { return isAltKey(window.modifiers); }},
    ctrlKey: {get: () => { return isCtrlKey(window.modifiers); }},
    metaKey: {get: () => { return isMetaKey(window.modifiers); }},
    shiftKey: {get: () => { return isShiftKey(window.modifiers); }},
    capsLock: {get: () => { return isCapsLock(window.modifiers); }},

    visable: {
      get() { return this._visible; },
      set(_: boolean) {
        if (this._visible !== _) { ((this._visible = _)) ? this.show() : this.hide(); }
      },
    },

    decorated: {
      get() { return this._decorated; },
      set(_: boolean) {
        if (this._decorated !== _) {
          this._decorated = _;
          if (this._id > 0) { glfw.setWindowAttrib(this._id, GLFWWindowAttribute.DECORATED, _); }
        }
      }
    },

    transparent: {
      get() { return this._transparent; },
      set(_: boolean) {
        if (this._transparent !== _) { this._transparent = _; }
      }
    },

    resizable: {
      get() { return this._resizable; },
      set(_: boolean) {
        if (this._resizable !== _) {
          this._resizable = _;
          if (this._id > 0) { glfw.setWindowAttrib(this._id, GLFWWindowAttribute.RESIZABLE, _); }
        }
      }
    },

    cursor: {
      get() { return this._cursor; },
      set(_: any) {
        switch (_) {
          case 'pointer': _ = GLFWStandardCursor.HAND; break;
          case 'text': _ = GLFWStandardCursor.IBEAM; break;
          case 'crosshair': _ = GLFWStandardCursor.CROSSHAIR; break;
          case 'e-resize':
          case 'w-resize':
          case 'ew-resize': _ = GLFWStandardCursor.HRESIZE; break;
          case 'n-resize':
          case 's-resize':
          case 'ns-resize': _ = GLFWStandardCursor.VRESIZE; break;
          case 'auto':
          case 'none':
          case 'grab':
          case 'grabbing':
          case 'default':
          default: _ = GLFWStandardCursor.ARROW; break;
        }
        if (this._cursor !== _) {
          this._cursor = _;
          if (this._id > 0) {
            switch (_) {
              case GLFWStandardCursor.ARROW:
              case GLFWStandardCursor.IBEAM:
              case GLFWStandardCursor.CROSSHAIR:
              case GLFWStandardCursor.HAND:
              case GLFWStandardCursor.HRESIZE:
              case GLFWStandardCursor.VRESIZE: glfw.setCursor(this._id, _); break;
              default: break;
            }
          }
        }
      }
    },

    style: {
      get() {
        // eslint-disable-next-line @typescript-eslint/no-this-alias
        const self = this;
        return {
          get width() { return self.width; },
          set width(_: number) { self.width = _; },
          get height() { return self.height; },
          set height(_: number) { self.height = _; },
          get cursor() { return self.cursor; },
          set cursor(_: any) { self.cursor = _; },
        };
      }
    },

    frameBufferWidth: {
      get() { return this.width; },
    },
    frameBufferHeight: {
      get() { return this.height; },
    },

    openGLClientAPI: {value: GLFWClientAPI.OPENGL},
    openGLProfile: {value: GLFWOpenGLProfile.ANY},
    openGLMajorVersion: {
      value: 4,
    },
    openGLMinorVersion: {
      value: 6,
    },
    openGLForwardCompat: {
      value: true,
    },
    openGLContextCreationAPI: {
      value: GLFWContextCreationAPI.EGL,
    },

    subscriptions: {
      value: new Subscription(),
    },
  });

  // Attatching functions

  window.setAttribute = function(name: any, value: any) {
    if (name in this) { (this as any)[name] = value; }
  };

  window.getBoundingClientRect = function() {
    return {
      x: 0,
      y: 0,
      width: this.width,
      height: this.height,
      left: 0,
      top: 0,
      right: this.width,
      bottom: this.height,
    };
  };

  window.show = function() {
    this.visible = true;
    this._id || this._create();
    glfw.showWindow(this._id);
    return this;
  };

  window.hide = function() {
    this.visible = false;
    this._id || this._create();
    glfw.hideWindow(this._id);
    return this;
  };

  window.destoryGLFWWindow = function() {
    const id = this._id;
    this._subscriptions.unsubscribe();
    if (id) {
      this._id = <any>undefined;
      glfw.destroyWindow(id);
      if (!this._forceNewWindow && rootWindow === this) { setImmediate(() => process.exit(0)); }
    }
  };

  try {
    // let root = null;
    // if (!window._forceNewWindow && rootWindow) { root = rootWindow.id; }
    // const monitor = window._monitor ? window._monitor.id : null;

    glfw.windowHint(GLFWWindowAttribute.SAMPLES, 4);
    glfw.windowHint(GLFWWindowAttribute.DOUBLEBUFFER, true);
    glfw.windowHint(GLFWWindowAttribute.FOCUSED, window.focused);
    glfw.windowHint(GLFWWindowAttribute.FOCUS_ON_SHOW, window.focused);
    glfw.windowHint(GLFWWindowAttribute.VISIBLE, window.visible);
    glfw.windowHint(GLFWWindowAttribute.DECORATED, window.decorated);
    glfw.windowHint(GLFWWindowAttribute.RESIZABLE, window.resizable);
    glfw.windowHint(GLFWWindowAttribute.TRANSPARENT_FRAMEBUFFER, window.transparent);

    glfw.windowHint(GLFWWindowAttribute.CLIENT_API, window.openGLClientAPI);
    glfw.windowHint(GLFWWindowAttribute.OPENGL_DEBUG_CONTEXT, window.debug);
    glfw.windowHint(GLFWWindowAttribute.OPENGL_PROFILE, window.openGLProfile);
    glfw.windowHint(GLFWWindowAttribute.CONTEXT_VERSION_MAJOR, window.openGLMajorVersion);
    glfw.windowHint(GLFWWindowAttribute.CONTEXT_VERSION_MINOR, window.openGLMinorVersion);
    glfw.windowHint(GLFWWindowAttribute.OPENGL_FORWARD_COMPAT, window.openGLForwardCompat);
    glfw.windowHint(GLFWWindowAttribute.CONTEXT_CREATION_API, window.openGLContextCreationAPI);

    const id = glfw.createWindow(window.width, window.height, window.title, null, null);

    window.id = id;

    glfw.setInputMode(window._id, GLFWInputMode.LOCK_KEY_MODS, true);
    glfw.setInputMode(id, GLFWInputMode.CURSOR, GLFW.CURSOR_NORMAL);
    glfw.makeContextCurrent(id);

    ({x: window._x, y: window._y} = glfw.getWindowPos(id));
    ({width: window._width, height: window._height} = glfw.getWindowSize(id));
    ({xscale: window._xscale, yscale: window._yscale} = glfw.getWindowContentScale(id));

    // not sure how to handle root window atm
    // !window._forceNewWindow && !rootWindow && (rootWindow = this);
    window._frameBufferWidth  = window._width * window._xscale;
    window._frameBufferHeight = window._height * window._yscale;
    window._subscriptions && window._subscriptions.unsubscribe();
    window._subscriptions = new Subscription();
    window.cursor         = window._cursor;

    glfw.swapInterval(window.swapInterval);
    glfw.swapBuffers(id);

    // FUCK - Argument of type 'DOMWindow' is not assignable to parameter of type 'GLFWDOMWindow'.
    // replaced with new event code and getting seg faults.......

    [dndEvents(window).subscribe(onGLFWDndEvent.bind(window)),
     mouseEvents(window).subscribe(onGLFWMouseEvent.bind(window)),
     wheelEvents(window).subscribe(onGLFWWheelEvent.bind(window)),
     windowEvents(window).subscribe(onGLFWWindowEvent.bind(window)),
     keyboardEvents(window).subscribe(onGLFWKeyboardEvent.bind(window)),
    ].forEach((subscription) => window._subscriptions.add(subscription));
    // const obser = dndEvents(window);
    // const smth  = onGLFWDndEvent.bind(window);
    // const sub   = obser.subscribe(smth);  // This line causes a segfault
    // window.subscriptions.add(sub);
    // window.subscriptions.add(dndEvents(window).subscribe(onGLFWDndEvent.bind(window)));
  } catch (e) {
    console.error('Error creating window:', e);
    window.destroyGLFWWindow();
    throw e;
  }

  return window;
}

function onGLFWDndEvent(this: DOMWindow, event: GLFWDndEvent) {
  dispatchGLFWEvent(this, event, this.Event);
}

function onGLFWMouseEvent(this: DOMWindow, event: GLFWMouseEvent) {
  this.mouseX   = event.x;
  this._mouseY  = event.y;
  this._buttons = event.buttons;
  let m         = this._modifiers;
  m             = event.altKey ? (m | GLFWModifierKey.MOD_ALT) : (m & ~GLFWModifierKey.MOD_ALT);
  m = event.ctrlKey ? (m | GLFWModifierKey.MOD_CONTROL) : (m & ~GLFWModifierKey.MOD_CONTROL);
  m = event.metaKey ? (m | GLFWModifierKey.MOD_SUPER) : (m & ~GLFWModifierKey.MOD_SUPER);
  m = event.shiftKey ? (m | GLFWModifierKey.MOD_SHIFT) : (m & ~GLFWModifierKey.MOD_SHIFT);
  m = event.capsLock ? (m | GLFWModifierKey.MOD_CAPS_LOCK) : (m & ~GLFWModifierKey.MOD_CAPS_LOCK);
  this._modifiers = m;
  dispatchGLFWEvent(this, event, this.MouseEvent);
}

function onGLFWWheelEvent(this: DOMWindow, event: GLFWWheelEvent) {
  this._scrollX += event.deltaX;
  this._scrollY += event.deltaY;
  dispatchGLFWEvent(this, event, this.WheelEvent);
}

function onGLFWWindowEvent(this: DOMWindow, event: GLFWWindowEvent) {
  this._x                 = event.x;
  this._y                 = event.y;
  this._width             = event.width;
  this._height            = event.height;
  this._xscale            = event.xscale;
  this._yscale            = event.yscale;
  this._focused           = event.focused;
  this._minimized         = event.minimized;
  this._maximized         = event.maximized;
  this._frameBufferWidth  = event.frameBufferWidth;
  this._frameBufferHeight = event.frameBufferHeight;
  this._devicePixelRatio =
    Math.min(this._frameBufferWidth / this._width, this._frameBufferHeight / this._height);
  dispatchGLFWEvent(this, event, this.Event);
  if (event.type === 'close') { this._destroyGLFWWindow(); }
}

function onGLFWKeyboardEvent(this: DOMWindow, event: GLFWKeyboardEvent) {
  let m = this._modifiers;
  m     = event.altKey ? (m | GLFWModifierKey.MOD_ALT) : (m & ~GLFWModifierKey.MOD_ALT);
  m     = event.ctrlKey ? (m | GLFWModifierKey.MOD_CONTROL) : (m & ~GLFWModifierKey.MOD_CONTROL);
  m     = event.metaKey ? (m | GLFWModifierKey.MOD_SUPER) : (m & ~GLFWModifierKey.MOD_SUPER);
  m     = event.shiftKey ? (m | GLFWModifierKey.MOD_SHIFT) : (m & ~GLFWModifierKey.MOD_SHIFT);
  m = event.capsLock ? (m | GLFWModifierKey.MOD_CAPS_LOCK) : (m & ~GLFWModifierKey.MOD_CAPS_LOCK);
  this._modifiers = m;
  dispatchGLFWEvent(this, event, this.KeyboardEvent);
}

function dispatchGLFWEvent(window: DOMWindow, glfwEvent: any, EventCtor: any) {
  const target = window._inputEventTarget || window.document;
  if (target && target.dispatchEvent) {
    target.dispatchEvent(asJSDOMEvent(EventCtor, glfwEvent, target));
  }
}

const {implSymbol} =
  ((global as any).idlUtils || require('jsdom/lib/jsdom/living/generated/utils'));

function asJSDOMEvent(EventCtor: any, glfwEvent: any, jsdomTarget: any) {
  glfwEvent.target     = jsdomTarget;
  const jsdomEvent     = new EventCtor(glfwEvent.type, glfwEvent);
  const jsdomEventImpl = jsdomEvent[implSymbol];
  for (const key in glfwEvent) {
    if (!key.startsWith('_')) {
      try {
        jsdomEventImpl[key] = glfwEvent[key];
      } catch (e) { /**/
      }
    }
  }
  return jsdomEvent;
}
