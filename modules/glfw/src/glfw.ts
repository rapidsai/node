// Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

export const isHeadless = typeof process.env.DISPLAY === 'undefined' ? true : false;
export const GLFW: any =
  require('bindings')(isHeadless ? 'rapidsai_glfw_eglheadless.node' : 'rapidsai_glfw_x11.node')
    .init();

export default GLFW;

export type GLFWmonitor      = number;
export type GLFWwindow       = number;
export type GLFWcursor       = number;
export type GLFWglproc       = number;
export type GLFWParentWindow = number|bigint|Buffer|ArrayBufferView|ArrayBufferView;

export interface GLFWVersion {
  major: number;
  minor: number;
  rev: number;
}

export interface GLFWScale {
  xscale: number;
  yscale: number;
}

export interface GLFWPosition {
  x: number;
  y: number;
}

export interface GLFWSize {
  width: number;
  height: number;
}

export interface GLFWRect {
  left: number;
  top: number;
  right: number;
  bottom: number;
}

export interface GLFWSizeLimits {
  minWidth: number;
  minHeight: number;
  maxWidth: number;
  maxHeight: number;
}

export interface GLFWPositionAndSize extends GLFWPosition, GLFWSize {}

export interface GLFWvidmode {
  /** The width, in screen coordinates, of the video mode.  */ width: number;
  /** The height, in screen coordinates, of the video mode. */ height: number;
  /** The bit depth of the red channel of the video mode.   */ redBits: number;
  /** The bit depth of the green channel of the video mode. */ greenBits: number;
  /** The bit depth of the blue channel of the video mode.  */ blueBits: number;
  /** The refresh rate, in Hz, of the video mode.           */ refreshRate: number;
}

export interface GLFWgammaramp {
  /** An array of value describing the response of the red channel.   */ red: number[];
  /** An array of value describing the response of the green channel. */ green: number[];
  /** An array of value describing the response of the blue channel.  */ blue: number[];
}

export interface GLFWimage {
  /** The width, in pixels, of this image.                                 */ width: number;
  /** The height, in pixels, of this image.                                */ height: number;
  /** The pixel data of this image, arranged left-to-right, top-to-bottom. */ pixels:
    ArrayBufferLike|ArrayBufferView;
}

export interface GLFWgamepadstate {
  /** The states of each gamepad button], `GLFW_PRESS` or `GLFW_RELEASE`. */ buttons: number[];
  /** The states of each gamepad axis], in the range -1.0 to 1.0 inclusive.  */ axes: number[];
}

// eslint-disable-next-line @typescript-eslint/no-namespace
export namespace glfw {

export const TRUE    = GLFW.TRUE;
export const FALSE   = GLFW.FALSE;
export const PRESS   = GLFW.PRESS;
export const REPEAT  = GLFW.REPEAT;
export const RELEASE = GLFW.RELEASE;

// export const init: () => void = GLFW.init;
export const terminate: () => void                           = GLFW.terminate;
export const initHint: (hint: number, value: number|boolean) => void = GLFW.initHint;
export const getVersion: () => GLFWVersion = GLFW.getVersion;
export const getVersionString: () => string = GLFW.getVersionString;
export const getError: () => Error | undefined = GLFW.getError;
export const getMonitors: () => GLFWmonitor[] = GLFW.getMonitors;
export const getPrimaryMonitor: () => GLFWmonitor  = GLFW.getPrimaryMonitor;
export const getMonitorPos: (monitor: GLFWmonitor) => GLFWPosition = GLFW.getMonitorPos;
export const getMonitorWorkarea:
  (monitor: GLFWmonitor) => GLFWPositionAndSize = GLFW.getMonitorWorkarea;
export const getMonitorPhysicalSize:
  (monitor: GLFWmonitor) => GLFWSize = GLFW.getMonitorPhysicalSize;
export const getMonitorContentScale:
  (monitor: GLFWmonitor) => GLFWScale               = GLFW.getMonitorContentScale;
export const getMonitorName: (monitor: GLFWmonitor) => string = GLFW.getMonitorName;
export const getVideoModes: (monitor: GLFWmonitor) => GLFWvidmode[] = GLFW.getVideoModes;
export const getVideoMode: (monitor: GLFWmonitor) => GLFWvidmode = GLFW.getVideoMode;
export const setGamma: (monitor: GLFWmonitor, gamma: number) => void = GLFW.setGamma;
export const getGammaRamp: (monitor: GLFWmonitor) => GLFWgammaramp     = GLFW.getGammaRamp;
export const setGammaRamp: (monitor: GLFWmonitor, rapi: GLFWgammaramp) => void = GLFW.setGammaRamp;
export const defaultWindowHints: () => void      = GLFW.defaultWindowHints;
export const windowHint: (hint: GLFWWindowAttribute,
                          value: number|boolean) => void = GLFW.windowHint;
export const windowHintString: (hint: string,
                                value: number|boolean) => void = GLFW.windowHintString;
export const createWindow: (width: number,
                            height: number,
                            title: string,
                            monitor?: GLFWmonitor|null,
                            root?: GLFWwindow|null) => GLFWwindow = GLFW.createWindow;
export const reparentWindow: (child: GLFWwindow,
                              parent: GLFWParentWindow,
                              targetX: number,
                              targetY: number) => void = GLFW.reparentWindow;
export const destroyWindow: (window: GLFWwindow) => void = GLFW.destroyWindow;
export const windowShouldClose: (window: GLFWwindow) => boolean = GLFW.windowShouldClose;
export const setWindowShouldClose: (window: GLFWwindow,
                                    shouldClose: boolean) => void = GLFW.setWindowShouldClose;
export const setWindowTitle: (window: GLFWwindow, title: string) => void = GLFW.setWindowTitle;
export const setWindowIcon: (window: GLFWwindow, icon: GLFWimage) => void = GLFW.setWindowIcon;
export const getWindowPos: (window: GLFWwindow) => GLFWPosition         = GLFW.getWindowPos;
export const setWindowPos: (window: GLFWwindow, position: GLFWPosition) => void = GLFW.setWindowPos;
export const getWindowSize: (window: GLFWwindow) => GLFWSize = GLFW.getWindowSize;
export const setWindowSizeLimits: (window: GLFWwindow,
                                   limits: GLFWSizeLimits) => void = GLFW.setWindowSizeLimits;
export const setWindowAspectRatio:
  (window: GLFWwindow, num: number, denom: number) => void       = GLFW.setWindowAspectRatio;
export const setWindowSize: (window: GLFWwindow, size: GLFWSize) => void = GLFW.setWindowSize;
export const getFramebufferSize: (window: GLFWwindow) => GLFWSize = GLFW.getFramebufferSize;
export const getWindowFrameSize: (window: GLFWwindow) => GLFWRect = GLFW.getWindowFrameSize;
export const getWindowContentScale: (window: GLFWwindow) => GLFWScale = GLFW.getWindowContentScale;
export const getWindowOpacity: (window: GLFWwindow) => number = GLFW.getWindowOpacity;
export const setWindowOpacity: (window: GLFWwindow,
                                opacity: number) => void = GLFW.setWindowOpacity;
export const iconifyWindow: (window: GLFWwindow) => void = GLFW.iconifyWindow;
export const restoreWindow: (window: GLFWwindow) => void = GLFW.restoreWindow;
export const maximizeWindow: (window: GLFWwindow) => void = GLFW.maximizeWindow;
export const showWindow: (window: GLFWwindow) => void = GLFW.showWindow;
export const hideWindow: (window: GLFWwindow) => void = GLFW.hideWindow;
export const focusWindow: (window: GLFWwindow) => void    = GLFW.focusWindow;
export const requestWindowAttention: (window: GLFWwindow) => void = GLFW.requestWindowAttention;
export const getWindowMonitor: (window: GLFWwindow) => GLFWmonitor = GLFW.getWindowMonitor;
export const setWindowMonitor: (window: GLFWwindow,
                                monitor: GLFWmonitor) => void  = GLFW.setWindowMonitor;
export const getWindowAttrib: (window: GLFWwindow,
                               attribute: GLFWWindowAttribute) => void = GLFW.getWindowAttrib;
export const setWindowAttrib: (window: GLFWwindow,
                               attribute: GLFWWindowAttribute,
                               value: number|boolean) => void = GLFW.setWindowAttrib;
export const pollEvents: () => void = GLFW.pollEvents;
export const waitEvents: () => void               = GLFW.waitEvents;
export const waitEventsTimeout: (timeout: number) => void = GLFW.waitEventsTimeout;
export const postEmptyEvent: () => void                       = GLFW.postEmptyEvent;
export const getInputMode: (window: GLFWwindow, mode: number) => number = GLFW.getInputMode;
export const setInputMode:
  (window: GLFWwindow, mode: number, value: number|boolean) => void = GLFW.setInputMode;
export const rawMouseMotionSupported: () => boolean      = GLFW.rawMouseMotionSupported;
export const getKeyName: (key: number, scancode: number) => string = GLFW.getKeyName;
export const getKeyScancode: (key: number) => number   = GLFW.getKeyScancode;
export const getKey: (window: GLFWwindow, key: number) => void    = GLFW.getKey;
export const getMouseButton: (window: GLFWwindow, button: number) => number = GLFW.getMouseButton;
export const getCursorPos: (window: GLFWwindow) => GLFWPosition         = GLFW.getCursorPos;
export const setCursorPos: (window: GLFWwindow, position: GLFWPosition) => void = GLFW.setCursorPos;
export const createCursor:
  (image: GLFWimage, x: number, y: number) => GLFWcursor = GLFW.createCursor;
export const createStandardCursor: (shape: number) => GLFWcursor = GLFW.createStandardCursor;
export const destroyCursor: (cursor: GLFWcursor) => void         = GLFW.destroyCursor;
export const setCursor: (window: GLFWwindow, cursor: GLFWcursor) => void = GLFW.setCursor;

export const joystickPresent: (joystickId: number) => boolean = GLFW.joystickPresent;
export const getJoystickAxes: (joystickId: number) => number[] = GLFW.getJoystickAxes;
export const getJoystickButtons: (joystickId: number) => number[] = GLFW.getJoystickButtons;
export const getJoystickHats: (joystickId: number) => number[] = GLFW.getJoystickHats;
export const getJoystickName: (joystickId: number) => string = GLFW.getJoystickName;
export const getJoystickGUID: (joystickId: number) => string = GLFW.getJoystickGUID;
export const joystickIsGamepad: (joystickId: number) => boolean = GLFW.joystickIsGamepad;
export const updateGamepadMappings: (mappings: string) => void = GLFW.updateGamepadMappings;
export const getGamepadName: (joystickId: number) => string = GLFW.getGamepadName;
export const getGamepadState: (joystickId: number) => GLFWgamepadstate = GLFW.getGamepadState;
export const setClipboardString: (window: GLFWwindow,
                                  value: string) => void = GLFW.setClipboardString;
export const getClipboardString: (window: GLFWwindow) => string = GLFW.getClipboardString;
export const getTime: () => number   = GLFW.getTime;
export const setTime: (time: number) => void = GLFW.setTime;
export const getTimerValue: () => number = GLFW.getTimerValue;
export const getTimerFrequency: () => number          = GLFW.getTimerFrequency;
export const makeContextCurrent: (window: GLFWwindow) => void = GLFW.makeContextCurrent;
export const getCurrentContext: () => GLFWwindow = GLFW.getCurrentContext;
export const swapBuffers: (window: GLFWwindow) => void = GLFW.swapBuffers;
export const swapInterval: (interval: number) => void = GLFW.swapInterval;
export const extensionSupported: (extension: string) => boolean = GLFW.extensionSupported;
export const getProcAddress: (procname: string) => GLFWglproc = GLFW.getProcAddress;
export const vulkanSupported: () => boolean    = GLFW.vulkanSupported;
export const getRequiredInstanceExtensions: () => string[] = GLFW.getRequiredInstanceExtensions;

export const setErrorCallback:
  (callback: null|
   ((error_code: number, description: string) => void)) => void = GLFW.setErrorCallback;
export const setMonitorCallback:
  (callback: null|
   ((monitor: GLFWmonitor, event: number) => void)) => void = GLFW.setMonitorCallback;
export const setJoystickCallback:
  (callback: null|((joystick: number, event: number) => void)) => void = GLFW.setJoystickCallback;
export const setWindowPosCallback:
  (window: GLFWwindow,
   callback: null|
   ((window: GLFWwindow, x: number, y: number) => void)) => void = GLFW.setWindowPosCallback;
export const setWindowSizeCallback:
  (window: GLFWwindow,
   callback: null|((window: GLFWwindow, width: number, height: number) => void)) => void =
    GLFW.setWindowSizeCallback;
export const setWindowCloseCallback:
  (window: GLFWwindow,
   callback: null|((window: GLFWwindow) => void)) => void = GLFW.setWindowCloseCallback;
export const setWindowRefreshCallback:
  (window: GLFWwindow,
   callback: null|((window: GLFWwindow) => void)) => void = GLFW.setWindowRefreshCallback;
export const setWindowFocusCallback:
  (window: GLFWwindow,
   callback: null|
   ((window: GLFWwindow, focused: boolean) => void)) => void = GLFW.setWindowFocusCallback;
export const setWindowIconifyCallback:
  (window: GLFWwindow,
   callback: null|
   ((window: GLFWwindow, iconified: boolean) => void)) => void = GLFW.setWindowIconifyCallback;
export const setWindowMaximizeCallback:
  (window: GLFWwindow,
   callback: null|
   ((window: GLFWwindow, maximized: boolean) => void)) => void = GLFW.setWindowMaximizeCallback;
export const setFramebufferSizeCallback:
  (window: GLFWwindow,
   callback: null|((window: GLFWwindow, width: number, height: number) => void)) => void =
    GLFW.setFramebufferSizeCallback;
export const setWindowContentScaleCallback:
  (window: GLFWwindow,
   callback: null|((window: GLFWwindow, xscale: number, yscale: number) => void)) => void =
    GLFW.setWindowContentScaleCallback;
export const setKeyCallback:
  (window: GLFWwindow,
   callback: null|
   ((window: GLFWwindow, key: number, scancode: number, action: number, mods: number) => void)) =>
    void = GLFW.setKeyCallback;
export const setCharCallback:
  (window: GLFWwindow,
   callback: null|((window: GLFWwindow, code: number) => void)) => void = GLFW.setCharCallback;
export const setCharModsCallback:
  (window: GLFWwindow,
   callback: null|
   ((window: GLFWwindow, code: number, mods: number) => void)) => void = GLFW.setCharModsCallback;
export const setMouseButtonCallback:
  (window: GLFWwindow,
   callback: null|((window: GLFWwindow, button: number, action: number, mods: number) => void)) =>
    void = GLFW.setMouseButtonCallback;
export const setCursorPosCallback:
  (window: GLFWwindow,
   callback: null|
   ((window: GLFWwindow, x: number, y: number) => void)) => void = GLFW.setCursorPosCallback;
export const setCursorEnterCallback:
  (window: GLFWwindow,
   callback: null|
   ((window: GLFWwindow, entered: number) => void)) => void = GLFW.setCursorEnterCallback;
export const setScrollCallback:
  (window: GLFWwindow,
   callback: null|((window: GLFWwindow, xoffset: number, yoffset: number) => void)) => void =
    GLFW.setScrollCallback;
export const setDropCallback:
  (window: GLFWwindow,
   callback: null|((window: GLFWwindow, items: string[]) => void)) => void = GLFW.setDropCallback;

// export const getInstanceProcAddress: () => void = GLFW.getInstanceProcAddress;
// export const getPhysicalDevicePresentationSupport: () => void =
// GLFW.getPhysicalDevicePresentationSupport; export const createWindowSurface: () => void =
// GLFW.createWindowSurface;
}

export const GLFWStandardCursor = {
  ARROW: glfw.createStandardCursor(GLFW.ARROW_CURSOR),
  IBEAM: glfw.createStandardCursor(GLFW.IBEAM_CURSOR),
  CROSSHAIR: glfw.createStandardCursor(GLFW.CROSSHAIR_CURSOR),
  HAND: glfw.createStandardCursor(GLFW.HAND_CURSOR),
  HRESIZE: glfw.createStandardCursor(GLFW.HRESIZE_CURSOR),
  VRESIZE: glfw.createStandardCursor(GLFW.VRESIZE_CURSOR),
};

export enum GLFWClientAPI
{
  NONE      = GLFW.NO_API,
  OPENGL    = GLFW.OPENGL_API,
  OPENGL_ES = GLFW.OPENGL_ES_API,
}

export enum GLFWOpenGLProfile
{
  ANY    = GLFW.OPENGL_ANY_PROFILE,
  CORE   = GLFW.OPENGL_CORE_PROFILE,
  COMPAT = GLFW.OPENGL_COMPAT_PROFILE,
}

export enum GLFWContextCreationAPI
{
  EGL    = GLFW.EGL_CONTEXT_API,
  NATIVE = GLFW.NATIVE_CONTEXT_API,
  OSMESA = GLFW.OSMESA_CONTEXT_API,
}

export enum GLFWWindowAttribute
{
  FOCUSED                  = GLFW.FOCUSED,
  ICONIFIED                = GLFW.ICONIFIED,
  RESIZABLE                = GLFW.RESIZABLE,
  VISIBLE                  = GLFW.VISIBLE,
  DECORATED                = GLFW.DECORATED,
  AUTO_ICONIFY             = GLFW.AUTO_ICONIFY,
  FLOATING                 = GLFW.FLOATING,
  MAXIMIZED                = GLFW.MAXIMIZED,
  CENTER_CURSOR            = GLFW.CENTER_CURSOR,
  TRANSPARENT_FRAMEBUFFER  = GLFW.TRANSPARENT_FRAMEBUFFER,
  HOVERED                  = GLFW.HOVERED,
  FOCUS_ON_SHOW            = GLFW.FOCUS_ON_SHOW,
  RED_BITS                 = GLFW.RED_BITS,
  GREEN_BITS               = GLFW.GREEN_BITS,
  BLUE_BITS                = GLFW.BLUE_BITS,
  ALPHA_BITS               = GLFW.ALPHA_BITS,
  DEPTH_BITS               = GLFW.DEPTH_BITS,
  STENCIL_BITS             = GLFW.STENCIL_BITS,
  ACCUM_RED_BITS           = GLFW.ACCUM_RED_BITS,
  ACCUM_GREEN_BITS         = GLFW.ACCUM_GREEN_BITS,
  ACCUM_BLUE_BITS          = GLFW.ACCUM_BLUE_BITS,
  ACCUM_ALPHA_BITS         = GLFW.ACCUM_ALPHA_BITS,
  AUX_BUFFERS              = GLFW.AUX_BUFFERS,
  STEREO                   = GLFW.STEREO,
  SAMPLES                  = GLFW.SAMPLES,
  SRGB_CAPABLE             = GLFW.SRGB_CAPABLE,
  REFRESH_RATE             = GLFW.REFRESH_RATE,
  DOUBLEBUFFER             = GLFW.DOUBLEBUFFER,
  CLIENT_API               = GLFW.CLIENT_API,
  CONTEXT_VERSION_MAJOR    = GLFW.CONTEXT_VERSION_MAJOR,
  CONTEXT_VERSION_MINOR    = GLFW.CONTEXT_VERSION_MINOR,
  CONTEXT_REVISION         = GLFW.CONTEXT_REVISION,
  CONTEXT_ROBUSTNESS       = GLFW.CONTEXT_ROBUSTNESS,
  OPENGL_FORWARD_COMPAT    = GLFW.OPENGL_FORWARD_COMPAT,
  OPENGL_DEBUG_CONTEXT     = GLFW.OPENGL_DEBUG_CONTEXT,
  OPENGL_PROFILE           = GLFW.OPENGL_PROFILE,
  CONTEXT_RELEASE_BEHAVIOR = GLFW.CONTEXT_RELEASE_BEHAVIOR,
  CONTEXT_NO_ERROR         = GLFW.CONTEXT_NO_ERROR,
  CONTEXT_CREATION_API     = GLFW.CONTEXT_CREATION_API,
  SCALE_TO_MONITOR         = GLFW.SCALE_TO_MONITOR,
  COCOA_RETINA_FRAMEBUFFER = GLFW.COCOA_RETINA_FRAMEBUFFER,
  COCOA_FRAME_NAME         = GLFW.COCOA_FRAME_NAME,
  COCOA_GRAPHICS_SWITCHING = GLFW.COCOA_GRAPHICS_SWITCHING,
  X11_CLASS_NAME           = GLFW.X11_CLASS_NAME,
  X11_INSTANCE_NAME        = GLFW.X11_INSTANCE_NAME,
}

export enum GLFWModifierKey
{
  MOD_ALT       = GLFW.MOD_ALT,
  MOD_SHIFT     = GLFW.MOD_SHIFT,
  MOD_SUPER     = GLFW.MOD_SUPER,
  MOD_CONTROL   = GLFW.MOD_CONTROL,
  MOD_NUM_LOCK  = GLFW.MOD_NUM_LOCK,
  MOD_CAPS_LOCK = GLFW.MOD_CAPS_LOCK,
}

export enum GLFWMouseButton
{
  MOUSE_BUTTON_1      = GLFW.MOUSE_BUTTON_1,
  MOUSE_BUTTON_2      = GLFW.MOUSE_BUTTON_2,
  MOUSE_BUTTON_3      = GLFW.MOUSE_BUTTON_3,
  MOUSE_BUTTON_4      = GLFW.MOUSE_BUTTON_4,
  MOUSE_BUTTON_5      = GLFW.MOUSE_BUTTON_5,
  MOUSE_BUTTON_6      = GLFW.MOUSE_BUTTON_6,
  MOUSE_BUTTON_7      = GLFW.MOUSE_BUTTON_7,
  MOUSE_BUTTON_8      = GLFW.MOUSE_BUTTON_8,
  MOUSE_BUTTON_LAST   = GLFW.MOUSE_BUTTON_LAST,
  MOUSE_BUTTON_LEFT   = GLFW.MOUSE_BUTTON_LEFT,
  MOUSE_BUTTON_RIGHT  = GLFW.MOUSE_BUTTON_RIGHT,
  MOUSE_BUTTON_MIDDLE = GLFW.MOUSE_BUTTON_MIDDLE,
}

export enum GLFWInputMode
{
  CURSOR               = GLFW.CURSOR,
  STICKY_KEYS          = GLFW.STICKY_KEYS,
  STICKY_MOUSE_BUTTONS = GLFW.STICKY_MOUSE_BUTTONS,
  LOCK_KEY_MODS        = GLFW.LOCK_KEY_MODS,
  RAW_MOUSE_MOTION     = GLFW.RAW_MOUSE_MOTION,
}

export enum GLFWKey
{
  KEY_UNKNOWN       = GLFW.KEY_UNKNOWN,
  KEY_SPACE         = GLFW.KEY_SPACE,
  KEY_APOSTROPHE    = GLFW.KEY_APOSTROPHE,
  KEY_COMMA         = GLFW.KEY_COMMA,
  KEY_MINUS         = GLFW.KEY_MINUS,
  KEY_PERIOD        = GLFW.KEY_PERIOD,
  KEY_SLASH         = GLFW.KEY_SLASH,
  KEY_0             = GLFW.KEY_0,
  KEY_1             = GLFW.KEY_1,
  KEY_2             = GLFW.KEY_2,
  KEY_3             = GLFW.KEY_3,
  KEY_4             = GLFW.KEY_4,
  KEY_5             = GLFW.KEY_5,
  KEY_6             = GLFW.KEY_6,
  KEY_7             = GLFW.KEY_7,
  KEY_8             = GLFW.KEY_8,
  KEY_9             = GLFW.KEY_9,
  KEY_SEMICOLON     = GLFW.KEY_SEMICOLON,
  KEY_EQUAL         = GLFW.KEY_EQUAL,
  KEY_A             = GLFW.KEY_A,
  KEY_B             = GLFW.KEY_B,
  KEY_C             = GLFW.KEY_C,
  KEY_D             = GLFW.KEY_D,
  KEY_E             = GLFW.KEY_E,
  KEY_F             = GLFW.KEY_F,
  KEY_G             = GLFW.KEY_G,
  KEY_H             = GLFW.KEY_H,
  KEY_I             = GLFW.KEY_I,
  KEY_J             = GLFW.KEY_J,
  KEY_K             = GLFW.KEY_K,
  KEY_L             = GLFW.KEY_L,
  KEY_M             = GLFW.KEY_M,
  KEY_N             = GLFW.KEY_N,
  KEY_O             = GLFW.KEY_O,
  KEY_P             = GLFW.KEY_P,
  KEY_Q             = GLFW.KEY_Q,
  KEY_R             = GLFW.KEY_R,
  KEY_S             = GLFW.KEY_S,
  KEY_T             = GLFW.KEY_T,
  KEY_U             = GLFW.KEY_U,
  KEY_V             = GLFW.KEY_V,
  KEY_W             = GLFW.KEY_W,
  KEY_X             = GLFW.KEY_X,
  KEY_Y             = GLFW.KEY_Y,
  KEY_Z             = GLFW.KEY_Z,
  KEY_LEFT_BRACKET  = GLFW.KEY_LEFT_BRACKET,
  KEY_BACKSLASH     = GLFW.KEY_BACKSLASH,
  KEY_RIGHT_BRACKET = GLFW.KEY_RIGHT_BRACKET,
  KEY_GRAVE_ACCENT  = GLFW.KEY_GRAVE_ACCENT,
  KEY_WORLD_1       = GLFW.KEY_WORLD_1,
  KEY_WORLD_2       = GLFW.KEY_WORLD_2,
  KEY_ESCAPE        = GLFW.KEY_ESCAPE,
  KEY_ENTER         = GLFW.KEY_ENTER,
  KEY_TAB           = GLFW.KEY_TAB,
  KEY_BACKSPACE     = GLFW.KEY_BACKSPACE,
  KEY_INSERT        = GLFW.KEY_INSERT,
  KEY_DELETE        = GLFW.KEY_DELETE,
  KEY_RIGHT         = GLFW.KEY_RIGHT,
  KEY_LEFT          = GLFW.KEY_LEFT,
  KEY_DOWN          = GLFW.KEY_DOWN,
  KEY_UP            = GLFW.KEY_UP,
  KEY_PAGE_UP       = GLFW.KEY_PAGE_UP,
  KEY_PAGE_DOWN     = GLFW.KEY_PAGE_DOWN,
  KEY_HOME          = GLFW.KEY_HOME,
  KEY_END           = GLFW.KEY_END,
  KEY_CAPS_LOCK     = GLFW.KEY_CAPS_LOCK,
  KEY_SCROLL_LOCK   = GLFW.KEY_SCROLL_LOCK,
  KEY_NUM_LOCK      = GLFW.KEY_NUM_LOCK,
  KEY_PRINT_SCREEN  = GLFW.KEY_PRINT_SCREEN,
  KEY_PAUSE         = GLFW.KEY_PAUSE,
  KEY_F1            = GLFW.KEY_F1,
  KEY_F2            = GLFW.KEY_F2,
  KEY_F3            = GLFW.KEY_F3,
  KEY_F4            = GLFW.KEY_F4,
  KEY_F5            = GLFW.KEY_F5,
  KEY_F6            = GLFW.KEY_F6,
  KEY_F7            = GLFW.KEY_F7,
  KEY_F8            = GLFW.KEY_F8,
  KEY_F9            = GLFW.KEY_F9,
  KEY_F10           = GLFW.KEY_F10,
  KEY_F11           = GLFW.KEY_F11,
  KEY_F12           = GLFW.KEY_F12,
  KEY_F13           = GLFW.KEY_F13,
  KEY_F14           = GLFW.KEY_F14,
  KEY_F15           = GLFW.KEY_F15,
  KEY_F16           = GLFW.KEY_F16,
  KEY_F17           = GLFW.KEY_F17,
  KEY_F18           = GLFW.KEY_F18,
  KEY_F19           = GLFW.KEY_F19,
  KEY_F20           = GLFW.KEY_F20,
  KEY_F21           = GLFW.KEY_F21,
  KEY_F22           = GLFW.KEY_F22,
  KEY_F23           = GLFW.KEY_F23,
  KEY_F24           = GLFW.KEY_F24,
  KEY_F25           = GLFW.KEY_F25,
  KEY_KP_0          = GLFW.KEY_KP_0,
  KEY_KP_1          = GLFW.KEY_KP_1,
  KEY_KP_2          = GLFW.KEY_KP_2,
  KEY_KP_3          = GLFW.KEY_KP_3,
  KEY_KP_4          = GLFW.KEY_KP_4,
  KEY_KP_5          = GLFW.KEY_KP_5,
  KEY_KP_6          = GLFW.KEY_KP_6,
  KEY_KP_7          = GLFW.KEY_KP_7,
  KEY_KP_8          = GLFW.KEY_KP_8,
  KEY_KP_9          = GLFW.KEY_KP_9,
  KEY_KP_DECIMAL    = GLFW.KEY_KP_DECIMAL,
  KEY_KP_DIVIDE     = GLFW.KEY_KP_DIVIDE,
  KEY_KP_MULTIPLY   = GLFW.KEY_KP_MULTIPLY,
  KEY_KP_SUBTRACT   = GLFW.KEY_KP_SUBTRACT,
  KEY_KP_ADD        = GLFW.KEY_KP_ADD,
  KEY_KP_ENTER      = GLFW.KEY_KP_ENTER,
  KEY_KP_EQUAL      = GLFW.KEY_KP_EQUAL,
  KEY_LEFT_SHIFT    = GLFW.KEY_LEFT_SHIFT,
  KEY_LEFT_CONTROL  = GLFW.KEY_LEFT_CONTROL,
  KEY_LEFT_ALT      = GLFW.KEY_LEFT_ALT,
  KEY_LEFT_SUPER    = GLFW.KEY_LEFT_SUPER,
  KEY_RIGHT_SHIFT   = GLFW.KEY_RIGHT_SHIFT,
  KEY_RIGHT_CONTROL = GLFW.KEY_RIGHT_CONTROL,
  KEY_RIGHT_ALT     = GLFW.KEY_RIGHT_ALT,
  KEY_RIGHT_SUPER   = GLFW.KEY_RIGHT_SUPER,
  KEY_MENU          = GLFW.KEY_MENU,
  KEY_LAST          = GLFW.KEY_LAST,
}
