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

import * as jsdom from 'jsdom';
import { Subscription } from 'rxjs';
import { Monitor } from '../monitor';
import { dndEvents, GLFWDndEvent } from '../events/dnd';
import { mouseEvents, GLFWMouseEvent } from '../events/mouse';
import { wheelEvents, GLFWWheelEvent } from '../events/wheel';
import { windowEvents, GLFWWindowEvent } from '../events/window';
import { keyboardEvents, GLFWKeyboardEvent } from '../events/keyboard';
import { isAltKey, isCtrlKey, isMetaKey, isShiftKey, isCapsLock } from '../events/event';
import { glfw, GLFW, GLFWwindow, GLFWInputMode, GLFWModifierKey, GLFWWindowAttribute, GLFWParentWindow, GLFWStandardCursor } from '../glfw';

export type GLFWDOMWindowOptions = {
    x?: number;
    y?: number;
    debug?: boolean;
    width?: number;
    height?: number;
    visible?: boolean;
    decorated?: boolean;
    resizable?: boolean;
    transparent?: boolean;
    devicePixelRatio?: number;
};

export interface GLFWDOMWindow extends jsdom.DOMWindow {
    _inputEventTarget: any;
}

export abstract class GLFWDOMWindow {

    constructor(options: GLFWDOMWindowOptions = {}) {
        this.init(options);
    }

    public init(options: GLFWDOMWindowOptions = {}) {
        Object.assign(this, {
            debug: false,
            _title: 'Untitled',
            _x: 0, _y: 0, _width: 800, _height: 600,
            _mouseX: 0, _mouseY: 0, _scrollX: 0, _scrollY: 0,
            _buttons: 0, _modifiers: 0,
            _xscale: 1, _yscale: 1, _devicePixelRatio: 1,
            _focused: false, _minimized: false, _maximized: false, _swapInterval: 1,
            _visible: true, _decorated: true, _transparent: false, _resizable: true,
            _subscriptions: new Subscription()
        }, options);

        const validatePropType = (name: keyof this, type: string) => {
            if (typeof this[name] !== type) {
                throw new TypeError(`options.${name} must be a ${type}`);
            }
        };

        ([
            'x', 'y', 'width', 'height', 'mouseX', 'mouseY',
            'scrollX', 'scrollY', 'buttons', 'modifiers',
            'xscale', 'yscale', 'devicePixelRatio'
        ] as (keyof this)[]).forEach((prop) => validatePropType(prop, 'number'));

        ([
            'debug', 'visible', 'decorated', 'resizable', 'transparent'
        ] as (keyof this)[]).forEach((prop) => validatePropType(prop, 'boolean'));

        this._frameBufferWidth = this._width;
        this._frameBufferHeight = this._height;

        return this;
    }

    protected _x = 0;
    public get x() { return this._x; }
    public set x(_: number) { this._x = _; }

    protected _y = 0;
    public get y() { return this._y; }
    public set y(_: number) { this._y = _; }

    protected _id: GLFWwindow = 0;
    public get id() { return this._id; }

    protected _title = '';
    public get title() { return this._title; }

    protected _event: any = undefined;
    public get event() { return this._event; }

    protected _cursor = GLFWStandardCursor.ARROW;
    public get cursor() { return this._cursor; }
    public set cursor(_: any) {
        switch (_) {
            case 'pointer':
                _ = GLFWStandardCursor.HAND;
                break;
            case 'text':
                _ = GLFWStandardCursor.IBEAM;
                break;
            case 'crosshair':
                _ = GLFWStandardCursor.CROSSHAIR;
                break;
            case 'e-resize':
            case 'w-resize':
            case 'ew-resize':
                _ = GLFWStandardCursor.HRESIZE;
                break;
            case 'n-resize':
            case 's-resize':
            case 'ns-resize':
                _ = GLFWStandardCursor.VRESIZE;
                break;
            case 'auto':
            case 'none':
            case 'grab':
            case 'grabbing':
            case 'default':
            default:
                _ = GLFWStandardCursor.ARROW;
                break;
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
                    case GLFWStandardCursor.VRESIZE:
                        glfw.setCursor(this._id, _);
                        break;
                    default: break;
                }
            }
        }
    }

    protected _width = 800;
    public get width() { return this._width; }
    public set width(_: number) { this._width = this._cssToNumber('width', _); }

    protected _height = 600;
    public get height() { return this._height; }
    public set height(_: number) { this._height = this._cssToNumber('height', _); }

    protected _xscale = 1;
    public get xscale() { return this._xscale; }

    protected _yscale = 1;
    public get yscale() { return this._yscale; }

    protected _mouseX = 0;
    public get mouseX() { return this._mouseX; }

    protected _mouseY = 0;
    public get mouseY() { return this._mouseY; }

    protected _scrollX = 0;
    public get scrollX() { return this._scrollX; }
    public set scrollX(_: number) { this._scrollX = _; }

    protected _scrollY = 0;
    public get scrollY() { return this._scrollY; }
    public set scrollY(_: number) { this._scrollY = _; }

    protected _buttons = 0;
    public get buttons() { return this._buttons; }

    protected _focused = false;
    public get focused() { return this._focused; }

    protected _minimized = false;
    public get minimized() { return this._minimized; }

    protected _maximized = false;
    public get maximized() { return this._maximized; }

    protected _swapInterval = 1;
    public get swapInterval() { return this._swapInterval; }
    public set swapInterval(_: number) {
        if (this._swapInterval !== _) {
            if (this._id > 0) {
                glfw.swapInterval(this._swapInterval = _);
            }
        }
    }

    protected _modifiers = 0;
    public get modifiers() { return this._modifiers; }

    public get altKey() { return isAltKey(this._modifiers); }
    public get ctrlKey() { return isCtrlKey(this._modifiers); }
    public get metaKey() { return isMetaKey(this._modifiers); }
    public get shiftKey() { return isShiftKey(this._modifiers); }
    public get capsLock() { return isCapsLock(this._modifiers); }

    protected _devicePixelRatio = 1;
    public get devicePixelRatio() { return this._devicePixelRatio; }
    public set devicePixelRatio(_: number) { this._devicePixelRatio = _; }

    public readonly debug = false;

    protected _visible = true;
    public get visible() { return this._visible; }
    public set visible(_: boolean) {
        if (this._visible !== _) {
            (this._visible = _) ? this.show() : this.hide();
        }
    }

    protected _decorated = true;
    public get decorated() { return this._decorated; }
    public set decorated(_: boolean) {
        if (this._decorated !== _) {
            this._decorated = _;
            if (this._id > 0) {
                glfw.setWindowAttrib(this._id, GLFWWindowAttribute.DECORATED, _);
            }
        }
    }

    protected _transparent = false;
    public get transparent() { return this._transparent; }
    public set transparent(_: boolean) {
        if (this._transparent !== _) {
            this._transparent = _;
        }
    }

    protected _resizable = true;
    public get resizable() { return this._resizable; }
    public set resizable(_: boolean) {
        if (this._resizable !== _) {
            this._resizable = _;
            if (this._id > 0) {
                glfw.setWindowAttrib(this._id, GLFWWindowAttribute.RESIZABLE, _);
            }
        }
    }

    public get style() {
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

    public setAttribute(name: any, value: any) {
        if (name in this) {
            (this as any)[name] = value;
        }
    }

    public getBoundingClientRect() {
        return {
            x: 0, y: 0, width: this.width, height: this.height,
            left: 0, top: 0, right: this.width, bottom: this.height,
        };
    }

    protected _frameBufferWidth = this._width;
    public get frameBufferWidth() { return this._frameBufferWidth; }

    protected _frameBufferHeight = this._height;
    public get frameBufferHeight() { return this._frameBufferHeight; }

    // @ts-ignore
    protected _subscriptions: Subscription;
    protected _monitor: Monitor | undefined;
    protected _rootWindow: GLFWDOMWindow | undefined;

    public destroyGLFWWindow() { this._destroyGLFWWindow(); }
    public show() {
        this.visible = true;
        this._id || this._create();
        glfw.showWindow(this._id);
        return this;
    }
    public hide() {
        this.visible = false;
        this._id || this._create();
        glfw.hideWindow(this._id);
        return this;
    }
    public reparent(parent: GLFWParentWindow, x = 0, y = 0) {
        if (this._id) {
            glfw.reparentWindow(this._id, parent, x, y);
        }
        return this;
    }

    protected _create() {
        if (this._id) { return; }
        try {
            const monitor = this._monitor ? this._monitor.id : null;
            const root = this._rootWindow ? this._rootWindow.id : null;

            glfw.windowHint(GLFWWindowAttribute.SAMPLES, 4);
            glfw.windowHint(GLFWWindowAttribute.DOUBLEBUFFER, true);
            glfw.windowHint(GLFWWindowAttribute.VISIBLE, this.visible);
            glfw.windowHint(GLFWWindowAttribute.DECORATED, this.decorated);
            glfw.windowHint(GLFWWindowAttribute.RESIZABLE, this.resizable);
            glfw.windowHint(GLFWWindowAttribute.OPENGL_DEBUG_CONTEXT, this.debug);
            glfw.windowHint(GLFWWindowAttribute.TRANSPARENT_FRAMEBUFFER, this.transparent);

            const id = glfw.createWindow(this.width, this.height, this.title, monitor, root);

            this._id = id;

            // glfw.setInputMode(this._id, GLFWInputMode.LOCK_KEY_MODS, true);
            glfw.setInputMode(id, GLFWInputMode.CURSOR, GLFW.CURSOR_NORMAL);
            glfw.makeContextCurrent(id);

            ({ x: this._x, y: this._y } = glfw.getWindowPos(id));
            ({ width: this._width, height: this._height } = glfw.getWindowSize(id));
            ({ xscale: this._xscale, yscale: this._yscale } = glfw.getWindowContentScale(id));

            this._rootWindow || (this._rootWindow = this);
            this._frameBufferWidth = this._width * this._xscale;
            this._frameBufferHeight = this._height * this._yscale;
            this._subscriptions && this._subscriptions.unsubscribe();
            this._subscriptions = new Subscription();
            this.cursor = this._cursor;

            glfw.swapInterval(this.swapInterval);
            glfw.swapBuffers(id);

            [
                dndEvents(this).subscribe(onGLFWDndEvent.bind(this)),
                mouseEvents(this).subscribe(onGLFWMouseEvent.bind(this)),
                wheelEvents(this).subscribe(onGLFWWheelEvent.bind(this)),
                windowEvents(this).subscribe(onGLFWWindowEvent.bind(this)),
                keyboardEvents(this).subscribe(onGLFWKeyboardEvent.bind(this)),
            ].forEach((subscription) => this._subscriptions.add(subscription));
        } catch (e) {
            console.error('Error creating window:', e);
            this._destroyGLFWWindow();
            throw e;
        }
    }

    protected _destroyGLFWWindow() {
        const id = this._id;
        this._subscriptions.unsubscribe();
        if (id) {
            this._id = <any>undefined;
            glfw.destroyWindow(id);
            if (this._rootWindow === this) {
                setImmediate(() => process.exit(0));
            }
        }
    }

    public [Symbol.toStringTag]: string;
    public inspect() { return this.toString(); }
    public [Symbol.for('nodejs.util.inspect.custom')]() { return this.toString(); }
    public toString() {
        return `${this[Symbol.toStringTag]} {
            id: ${this.id},
            x: ${this.x},
            y: ${this.y},
            width: ${this.width},
            height: ${this.height}
        }`
    }

    protected _gl: any = null;
    protected get _clearMask() { return this._gl ? this._gl._clearMask : 0; }
    protected set _clearMask(_: any) { this._gl && (this._gl._clearMask = _); }

    protected _cssToNumber(prop: keyof this, value: any): number {
        switch (typeof value) {
            case 'number': return value;
            case 'string': {
                if (value.endsWith('px')) {
                    return +(value.slice(0, -1));
                } else if (value.endsWith('%')) {
                    return +this[prop] * (+value.slice(0, -1) / 100);
                }
            }
        }
        return ((value = +value) !== value) ? +this[prop] : value;
    }
}

GLFWDOMWindow.prototype[Symbol.toStringTag] = 'GLFWDOMWindow';

defineDOMEventListenerProperties(GLFWDOMWindow.prototype, [
    'onblur', 'onfocus', 'onmove', 'onresize', 'onkeyup', 'onkeydown',
    'onmousedown', 'onmouseup', 'onmousemove', 'onmouseenter', 'onmouseleave', 'onwheel',
]);

defineDOMElementPropertyAliases(GLFWDOMWindow.prototype, [
    { name: 'y', aliases: ['screenY', 'screenTop'] },
    { name: 'x', aliases: ['screenX', 'screenLeft'] },
    { name: 'scrollX', aliases: ['scrollLeft', 'pageXOffset'] },
    { name: 'scrollY', aliases: ['scrollTop', 'pageYOffset'] },
    { name: 'onwheel', aliases: ['onscroll', 'onmousewheel'] },
    { name: 'width', aliases: ['clientWidth', 'innerWidth', 'offsetWidth'] },
    { name: 'height', aliases: ['clientHeight', 'innerHeight', 'offsetHeight'] },
]);

function onGLFWDndEvent(this: GLFWDOMWindow, event: GLFWDndEvent) {
    dispatchGLFWEvent(this, event, this.Event);
}

function onGLFWMouseEvent(this: GLFWDOMWindow, event: GLFWMouseEvent) {
    this._mouseX = event.x;
    this._mouseY = event.y;
    this._buttons = event.buttons;
    let m = this._modifiers;
    m = event.altKey ? (m | GLFWModifierKey.MOD_ALT) : (m & ~GLFWModifierKey.MOD_ALT);
    m = event.ctrlKey ? (m | GLFWModifierKey.MOD_CONTROL) : (m & ~GLFWModifierKey.MOD_CONTROL);
    m = event.metaKey ? (m | GLFWModifierKey.MOD_SUPER) : (m & ~GLFWModifierKey.MOD_SUPER);
    m = event.shiftKey ? (m | GLFWModifierKey.MOD_SHIFT) : (m & ~GLFWModifierKey.MOD_SHIFT);
    m = event.capsLock ? (m | GLFWModifierKey.MOD_CAPS_LOCK) : (m & ~GLFWModifierKey.MOD_CAPS_LOCK);
    this._modifiers = m;
    dispatchGLFWEvent(this, event, this.MouseEvent);
}

function onGLFWWheelEvent(this: GLFWDOMWindow, event: GLFWWheelEvent) {
    this._scrollX += event.deltaX;
    this._scrollY += event.deltaY;
    dispatchGLFWEvent(this, event, this.WheelEvent);
}

function onGLFWWindowEvent(this: GLFWDOMWindow, event: GLFWWindowEvent) {
    this._x = event.x;
    this._y = event.y;
    this._width = event.width;
    this._height = event.height;
    this._xscale = event.xscale;
    this._yscale = event.yscale;
    this._focused = event.focused;
    this._minimized = event.minimized;
    this._maximized = event.maximized;
    this._frameBufferWidth = event.frameBufferWidth;
    this._frameBufferHeight = event.frameBufferHeight;
    this._devicePixelRatio = this._frameBufferWidth / this._width;
    dispatchGLFWEvent(this, event, this.Event);
    if (event.type === 'close') {
        this._destroyGLFWWindow();
    }
}

function onGLFWKeyboardEvent(this: GLFWDOMWindow, event: GLFWKeyboardEvent) {
    let m = this._modifiers;
    m = event.altKey ? (m | GLFWModifierKey.MOD_ALT) : (m & ~GLFWModifierKey.MOD_ALT);
    m = event.ctrlKey ? (m | GLFWModifierKey.MOD_CONTROL) : (m & ~GLFWModifierKey.MOD_CONTROL);
    m = event.metaKey ? (m | GLFWModifierKey.MOD_SUPER) : (m & ~GLFWModifierKey.MOD_SUPER);
    m = event.shiftKey ? (m | GLFWModifierKey.MOD_SHIFT) : (m & ~GLFWModifierKey.MOD_SHIFT);
    m = event.capsLock ? (m | GLFWModifierKey.MOD_CAPS_LOCK) : (m & ~GLFWModifierKey.MOD_CAPS_LOCK);
    this._modifiers = m;
    dispatchGLFWEvent(this, event, this.KeyboardEvent);
}

const { implSymbol } = ((global as any).idlUtils || require("jsdom/lib/jsdom/living/generated/utils"));

function dispatchGLFWEvent(window: GLFWDOMWindow, glfwEvent: any, EventCtor: any) {
    const target = window._inputEventTarget || window.document;
    if (target && target.dispatchEvent) {
        target.dispatchEvent(asJSDOMEvent(EventCtor, glfwEvent, target));
    }
}

function asJSDOMEvent(EventCtor: any, glfwEvent: any, jsdomTarget: any) {
    glfwEvent.target = jsdomTarget;
    const jsdomEvent = new EventCtor(glfwEvent.type, glfwEvent);
    const jsdomEventImpl = jsdomEvent[implSymbol];
    for (const key in glfwEvent) {
        if (!key.startsWith('_')) {
            try { jsdomEventImpl[key] = glfwEvent[key]; } catch (e) {}
        }
    }
    return jsdomEvent;
}

function defineDOMEventListenerProperties(proto: any, propertyNames: string[]) {
    propertyNames.forEach((name) => {
        const type = name.slice(2);
        const pname = `_${name}Listener`;
        Object.defineProperty(proto, name, {
            get() { return this[pname] || undefined; },
            set(listener: (...args: any[]) => any) {
                if (typeof this[pname] === 'function') {
                    this.removeEventListener(type, this[pname]);
                }
                if (typeof listener === 'function') {
                    this[pname] = (e: any) => {
                        e.preventDefault();
                        e.stopImmediatePropagation();
                        listener(e);
                    };
                    this.addEventListener(type, this[pname], true);
                }
            }
        });
    });
}

function defineDOMElementPropertyAliases(proto: any, aliases: { name: string, aliases: string[] }[]) {
    aliases.forEach(({ name, aliases = [] }) => {
        const descriptor = Object.getOwnPropertyDescriptor(proto, name);
        if (descriptor) {
            descriptor.get = ((get) => get && function (this: any) {
                return get.call(this);
            })(descriptor.get);
            descriptor.set = ((set) => set && function (this: any, _: any) {
                return set.call(this, _);
            })(descriptor.set);
            aliases.forEach((alias) => Object.defineProperty(proto, alias, descriptor));
        }
    });
}
