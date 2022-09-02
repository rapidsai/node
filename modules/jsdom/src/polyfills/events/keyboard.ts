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

import {glfw, GLFWKey} from '@rapidsai/glfw';
import {GLFWInputMode} from '@rapidsai/glfw';
import {DOMWindow} from 'jsdom';
import {Observable} from 'rxjs';
import {merge as mergeObservables} from 'rxjs';
import {filter, flatMap, map, mergeAll, publish, refCount, withLatestFrom} from 'rxjs/operators';

import {
  GLFWEvent,
  isAltKey,
  isCapsLock,
  isCtrlKey,
  isMetaKey,
  isShiftKey,
  windowCallbackAsObservable
} from './event';

export function keyboardEvents(window: DOMWindow) {
  const keys        = keyUpdates(window);
  const specialKeys = keys.pipe(filter(isSpecialKey)).pipe(flatMap(function*(keyEvt) {
    yield keyEvt;
    // Also yield Delete 'keydown' events as 'keypress'
    if (keyEvt.key === 'Delete' && keyEvt.type === 'keydown') {
      yield keyEvt.asCharacter(127, 'keypress');
    }
  }));
  const characterKeys =
    keys.pipe(filter(isCharacterKey), (charKeys) => characterUpdates(window, charKeys));

  return mergeObservables(specialKeys, characterKeys);
}

function keyUpdates(window: DOMWindow) {
  return windowCallbackAsObservable(glfw.setKeyCallback, window)
    .pipe(map(([, ...rest]) => GLFWKeyboardEvent.fromKeyEvent(window, ...rest)))
    .pipe(publish(), refCount());
}

function characterUpdates(window: DOMWindow, charKeys: Observable<GLFWKeyboardEvent>) {
  const charCodes = windowCallbackAsObservable(glfw.setCharCallback, window)
                      .pipe(map(([, charCode]) => charCode), publish(), refCount());
  return charCodes
    .pipe(withLatestFrom(charKeys,
                         function*(charCode, keyEvt) {
                           yield keyEvt;
                           // Also yield 'keydown' events as 'keypress'
                           yield keyEvt.asCharacter(charCode, 'keypress');
                           // GLFW doesn't dispatch keyup for character keys, so dispatch our own
                           yield keyEvt.asCharacter(0, 'keyup');
                         }))
    .pipe(mergeAll());
}

function isCharacterKey(evt: GLFWKeyboardEvent) { return !isSpecialKey(evt); }

function isSpecialKey(evt: GLFWKeyboardEvent) {
  switch (evt._rawKey) {
    // GLFW dispatches spacebar as both a keyboard and character input event,
    // but glfw.getKeyName() doesn't return a name for it.
    case GLFWKey.KEY_SPACE: return false;
    // Arrow keys are special keys
    // Modifier keys are special keys
    case GLFWKey.KEY_UP:
    case GLFWKey.KEY_DOWN:
    case GLFWKey.KEY_LEFT:
    case GLFWKey.KEY_RIGHT:
    case GLFWKey.KEY_END:
    case GLFWKey.KEY_HOME:
    case GLFWKey.KEY_PAGE_UP:
    case GLFWKey.KEY_PAGE_DOWN:
    case GLFWKey.KEY_DELETE:
    case GLFWKey.KEY_BACKSPACE:
    case GLFWKey.KEY_CAPS_LOCK:
    case GLFWKey.KEY_LEFT_ALT:
    case GLFWKey.KEY_RIGHT_ALT:
    case GLFWKey.KEY_LEFT_CONTROL:
    case GLFWKey.KEY_RIGHT_CONTROL:
    case GLFWKey.KEY_LEFT_SUPER:
    case GLFWKey.KEY_RIGHT_SUPER:
    case GLFWKey.KEY_LEFT_SHIFT:
    case GLFWKey.KEY_RIGHT_SHIFT: return true;
    default:
      // If GLFW didn't return a key name, it's a special key
      return !evt._rawName || evt.key === 'Unidentified';
  }
}

export class GLFWKeyboardEvent extends GLFWEvent {
  public static fromDOMEvent(window: DOMWindow, event: KeyboardEvent) {
    const evt     = new GLFWKeyboardEvent(event.type);
    evt.target    = window;
    evt._key      = event.key;
    evt._code     = event.code;
    evt._which    = event.keyCode;
    evt._charCode = event.charCode;
    evt._repeat   = event.repeat;
    evt._altKey   = event.altKey;
    evt._ctrlKey  = event.ctrlKey;
    evt._metaKey  = event.metaKey;
    evt._shiftKey = event.shiftKey;
    evt._capsLock = false;
    if (event.getModifierState) { evt._capsLock = event.getModifierState('CapsLock'); }
    return evt;
  }
  public static fromKeyEvent(
    window: DOMWindow, key: number, scancode: number, action: number, modifiers: number) {
    const down = action !== glfw.RELEASE;
    const name = glfw.getKeyName(key, scancode);
    const evt  = new GLFWKeyboardEvent(action === glfw.RELEASE ? 'keyup' : 'keydown');

    evt._rawKey      = key;
    evt._rawName     = name;
    evt._rawScanCode = scancode;

    evt.target  = window;
    evt._repeat = action === glfw.REPEAT;

    evt._key      = glfwKeyToKey[key] || name || 'Unidentified';
    evt._code     = glfwKeyToCode[key] || (name && `Key${name.toUpperCase()}`) || 'Unidentified';
    evt._which    = glfwKeyToKeyCode[key] || key;
    evt._charCode = 0;

    const isCaps  = key === GLFWKey.KEY_CAPS_LOCK;
    const isAlt   = !isCaps && (key === GLFWKey.KEY_LEFT_ALT || key === GLFWKey.KEY_RIGHT_ALT);
    const isShift = !isAlt && (key === GLFWKey.KEY_LEFT_SHIFT || key === GLFWKey.KEY_RIGHT_SHIFT);
    const isSuper = !isShift && (key === GLFWKey.KEY_LEFT_SUPER || key === GLFWKey.KEY_RIGHT_SUPER);
    const isControl =
      !isSuper && (key === GLFWKey.KEY_LEFT_CONTROL || key === GLFWKey.KEY_RIGHT_CONTROL);

    if (!glfw.getInputMode(window.id, GLFWInputMode.LOCK_KEY_MODS)) {
      evt._capsLock = down ? isCaps || window.capsLock : !isCaps && window.capsLock;
    } else {
      evt._capsLock = isCaps ? window.capsLock ? !down : down : isCapsLock(modifiers);
    }

    if (down) {
      evt._altKey   = isAlt || isAltKey(modifiers);
      evt._ctrlKey  = isControl || isCtrlKey(modifiers);
      evt._metaKey  = isSuper || isMetaKey(modifiers);
      evt._shiftKey = isShift || isShiftKey(modifiers);
    } else {
      evt._altKey   = !isAlt && isAltKey(modifiers);
      evt._ctrlKey  = !isControl && isCtrlKey(modifiers);
      evt._metaKey  = !isSuper && isMetaKey(modifiers);
      evt._shiftKey = !isShift && isShiftKey(modifiers);
    }

    return evt;
  }

  public asCharacter(charCode: number,
                     type              = this.type,
                     modifiers: number = this.target.modifiers) {
    if (this._rawKey === GLFWKey.KEY_DELETE) { charCode = 127; }
    const evt        = new GLFWKeyboardEvent(type);
    evt.target       = this.target;
    evt._charCode    = charCode;
    evt._which       = charCode || this._which;
    evt._rawKey      = this._rawKey;
    evt._rawName     = this._rawName;
    evt._rawScanCode = this._rawScanCode;
    evt._key         = (charCode && String.fromCharCode(charCode)) || this._key;
    evt._repeat      = this._repeat || this.type === 'keypress';
    evt._code        = glfwKeyToCode[this._rawKey] || `Key${this._rawName.toUpperCase()}`;
    evt._altKey      = isAltKey(modifiers) || this._altKey;
    evt._ctrlKey     = isCtrlKey(modifiers) || this._ctrlKey;
    evt._metaKey     = isMetaKey(modifiers) || this._metaKey;
    evt._shiftKey    = isShiftKey(modifiers) || this._shiftKey;
    evt._capsLock    = isCapsLock(modifiers) || this._capsLock;
    return evt;
  }

  public isComposing = false;

  private _key                   = '';
  private _code                  = '';
  private _which                 = 0;
  private _repeat                = false;
  private _altKey                = false;
  private _ctrlKey               = false;
  private _metaKey               = false;
  private _shiftKey              = false;
  private _capsLock              = false;
  private _charCode: number|null = null;

  public _rawKey      = 0;
  public _rawName     = '';
  public _rawScanCode = 0;
  public get key() { return this._key; }
  public get code() { return this._code; }
  public get which() { return this._which; }
  public get repeat() { return this._repeat; }
  public get keyCode() { return this._which; }
  public get charCode() { return this._charCode; }

  public get altKey() { return this._altKey; }
  public get ctrlKey() { return this._ctrlKey; }
  public get metaKey() { return this._metaKey; }
  public get shiftKey() { return this._shiftKey; }
  public get capsLock() { return this._capsLock; }
}

// Map of `GLFWKey` to `KeyboardEvent.prototype.code`:
const glfwKeyToCode: any = {
  [GLFWKey.KEY_APOSTROPHE]: 'Quote',
  [GLFWKey.KEY_BACKSLASH]: 'Backslash',
  [GLFWKey.KEY_BACKSPACE]: 'Backspace',
  [GLFWKey.KEY_CAPS_LOCK]: 'CapsLock',
  [GLFWKey.KEY_COMMA]: 'Comma',
  [GLFWKey.KEY_DELETE]: 'Delete',
  [GLFWKey.KEY_DOWN]: 'ArrowDown',
  [GLFWKey.KEY_END]: 'End',
  [GLFWKey.KEY_ENTER]: 'Enter',
  [GLFWKey.KEY_EQUAL]: 'Equal',
  [GLFWKey.KEY_ESCAPE]: 'Escape',
  [GLFWKey.KEY_F10]: 'F10',
  [GLFWKey.KEY_F11]: 'F11',
  [GLFWKey.KEY_F12]: 'F12',
  [GLFWKey.KEY_F13]: 'F13',
  [GLFWKey.KEY_F14]: 'F14',
  [GLFWKey.KEY_F15]: 'F15',
  [GLFWKey.KEY_F16]: 'F16',
  [GLFWKey.KEY_F17]: 'F17',
  [GLFWKey.KEY_F18]: 'F18',
  [GLFWKey.KEY_F19]: 'F19',
  [GLFWKey.KEY_F1]: 'F1',
  [GLFWKey.KEY_F20]: 'F20',
  [GLFWKey.KEY_F21]: 'F21',
  [GLFWKey.KEY_F22]: 'F22',
  [GLFWKey.KEY_F23]: 'F23',
  [GLFWKey.KEY_F24]: 'F24',
  [GLFWKey.KEY_F25]: 'F25',
  [GLFWKey.KEY_F2]: 'F2',
  [GLFWKey.KEY_F3]: 'F3',
  [GLFWKey.KEY_F4]: 'F4',
  [GLFWKey.KEY_F5]: 'F5',
  [GLFWKey.KEY_F6]: 'F6',
  [GLFWKey.KEY_F7]: 'F7',
  [GLFWKey.KEY_F8]: 'F8',
  [GLFWKey.KEY_F9]: 'F9',
  [GLFWKey.KEY_GRAVE_ACCENT]: 'Backquote',
  [GLFWKey.KEY_HOME]: 'Home',
  [GLFWKey.KEY_INSERT]: 'Insert',
  [GLFWKey.KEY_KP_0]: 'Numpad0',
  [GLFWKey.KEY_KP_1]: 'Numpad1',
  [GLFWKey.KEY_KP_2]: 'Numpad2',
  [GLFWKey.KEY_KP_3]: 'Numpad3',
  [GLFWKey.KEY_KP_4]: 'Numpad4',
  [GLFWKey.KEY_KP_5]: 'Numpad5',
  [GLFWKey.KEY_KP_6]: 'Numpad6',
  [GLFWKey.KEY_KP_7]: 'Numpad7',
  [GLFWKey.KEY_KP_8]: 'Numpad8',
  [GLFWKey.KEY_KP_9]: 'Numpad9',
  [GLFWKey.KEY_KP_ADD]: 'NumpadAdd',
  [GLFWKey.KEY_KP_DECIMAL]: 'NumpadDecimal',
  [GLFWKey.KEY_KP_DIVIDE]: 'NumpadDivide',
  [GLFWKey.KEY_KP_ENTER]: 'NumpadEnter',
  [GLFWKey.KEY_KP_EQUAL]: 'NumpadEqual',
  [GLFWKey.KEY_KP_MULTIPLY]: 'NumpadMultiply',
  [GLFWKey.KEY_KP_SUBTRACT]: 'NumpadSubtract',
  [GLFWKey.KEY_LEFT]: 'ArrowLeft',
  [GLFWKey.KEY_LEFT_ALT]: 'AltLeft',
  [GLFWKey.KEY_LEFT_BRACKET]: 'BracketLeft',
  [GLFWKey.KEY_LEFT_CONTROL]: 'ControlLeft',
  [GLFWKey.KEY_LEFT_SHIFT]: 'ShiftLeft',
  [GLFWKey.KEY_LEFT_SUPER]: 'MetaLeft',
  [GLFWKey.KEY_MENU]: 'Menu',
  [GLFWKey.KEY_MINUS]: 'Minus',
  [GLFWKey.KEY_NUM_LOCK]: 'NumLock',
  [GLFWKey.KEY_PAGE_DOWN]: 'PageDown',
  [GLFWKey.KEY_PAGE_UP]: 'PageUp',
  [GLFWKey.KEY_PAUSE]: 'Pause',
  [GLFWKey.KEY_PERIOD]: 'Period',
  [GLFWKey.KEY_PRINT_SCREEN]: 'PrintScreen',
  [GLFWKey.KEY_RIGHT]: 'ArrowRight',
  [GLFWKey.KEY_RIGHT_ALT]: 'AltRight',
  [GLFWKey.KEY_RIGHT_BRACKET]: 'BracketRight',
  [GLFWKey.KEY_RIGHT_CONTROL]: 'ControlRight',
  [GLFWKey.KEY_RIGHT_SHIFT]: 'ShiftRight',
  [GLFWKey.KEY_RIGHT_SUPER]: 'MetaRight',
  [GLFWKey.KEY_SCROLL_LOCK]: 'ScrollLock',
  [GLFWKey.KEY_SEMICOLON]: 'Semicolon',
  [GLFWKey.KEY_SLASH]: 'Slash',
  [GLFWKey.KEY_SPACE]: 'Space',
  [GLFWKey.KEY_TAB]: 'Tab',
  [GLFWKey.KEY_UP]: 'ArrowUp',
};

// Map of `GLFWKey` to `KeyboardEvent.prototype.key`:
const glfwKeyToKey: any = Object.assign({}, glfwKeyToCode, {
  [GLFWKey.KEY_SPACE]: ' ',
  [GLFWKey.KEY_LEFT_ALT]: 'Alt',
  [GLFWKey.KEY_LEFT_CONTROL]: 'Control',
  [GLFWKey.KEY_LEFT_SHIFT]: 'Shift',
  [GLFWKey.KEY_LEFT_SUPER]: 'Super',
  [GLFWKey.KEY_RIGHT_ALT]: 'Alt',
  [GLFWKey.KEY_RIGHT_CONTROL]: 'Control',
  [GLFWKey.KEY_RIGHT_SHIFT]: 'Shift',
  [GLFWKey.KEY_RIGHT_SUPER]: 'Super',
});

// Map of `GLFWKey` to `KeyboardEvent.prototype.which`:
const glfwKeyToKeyCode: any = {
  [GLFWKey.KEY_APOSTROPHE]: 222,
  [GLFWKey.KEY_BACKSLASH]: 220,
  [GLFWKey.KEY_BACKSPACE]: 8,
  [GLFWKey.KEY_CAPS_LOCK]: 20,
  [GLFWKey.KEY_COMMA]: 188,
  [GLFWKey.KEY_DELETE]: 46,
  [GLFWKey.KEY_DOWN]: 40,
  [GLFWKey.KEY_END]: 35,
  [GLFWKey.KEY_ENTER]: 13,
  [GLFWKey.KEY_EQUAL]: 187,
  [GLFWKey.KEY_ESCAPE]: 27,
  [GLFWKey.KEY_F10]: 121,
  [GLFWKey.KEY_F11]: 122,
  [GLFWKey.KEY_F12]: 123,
  [GLFWKey.KEY_F13]: 123,
  [GLFWKey.KEY_F14]: 123,
  [GLFWKey.KEY_F15]: 123,
  [GLFWKey.KEY_F16]: 123,
  [GLFWKey.KEY_F17]: 123,
  [GLFWKey.KEY_F18]: 123,
  [GLFWKey.KEY_F19]: 123,
  [GLFWKey.KEY_F1]: 112,
  [GLFWKey.KEY_F20]: 123,
  [GLFWKey.KEY_F21]: 123,
  [GLFWKey.KEY_F22]: 123,
  [GLFWKey.KEY_F23]: 123,
  [GLFWKey.KEY_F24]: 123,
  [GLFWKey.KEY_F25]: 123,
  [GLFWKey.KEY_F2]: 113,
  [GLFWKey.KEY_F3]: 114,
  [GLFWKey.KEY_F4]: 115,
  [GLFWKey.KEY_F5]: 116,
  [GLFWKey.KEY_F6]: 117,
  [GLFWKey.KEY_F7]: 118,
  [GLFWKey.KEY_F8]: 119,
  [GLFWKey.KEY_F9]: 120,
  [GLFWKey.KEY_GRAVE_ACCENT]: 192,
  [GLFWKey.KEY_HOME]: 36,
  [GLFWKey.KEY_INSERT]: 45,
  [GLFWKey.KEY_KP_0]: 96,
  [GLFWKey.KEY_KP_1]: 97,
  [GLFWKey.KEY_KP_2]: 98,
  [GLFWKey.KEY_KP_3]: 99,
  [GLFWKey.KEY_KP_4]: 100,
  [GLFWKey.KEY_KP_5]: 101,
  [GLFWKey.KEY_KP_6]: 102,
  [GLFWKey.KEY_KP_7]: 103,
  [GLFWKey.KEY_KP_8]: 104,
  [GLFWKey.KEY_KP_9]: 105,
  [GLFWKey.KEY_KP_ADD]: 107,
  [GLFWKey.KEY_KP_DECIMAL]: 110,
  [GLFWKey.KEY_KP_DIVIDE]: 111,
  [GLFWKey.KEY_KP_ENTER]: 13,
  [GLFWKey.KEY_KP_EQUAL]: 187,
  [GLFWKey.KEY_KP_MULTIPLY]: 106,
  [GLFWKey.KEY_KP_SUBTRACT]: 109,
  [GLFWKey.KEY_LEFT]: 37,
  [GLFWKey.KEY_LEFT_ALT]: 18,
  [GLFWKey.KEY_LEFT_BRACKET]: 219,
  [GLFWKey.KEY_LEFT_CONTROL]: 17,
  [GLFWKey.KEY_LEFT_SHIFT]: 16,
  [GLFWKey.KEY_LEFT_SUPER]: 91,
  [GLFWKey.KEY_MENU]: 18,
  [GLFWKey.KEY_MINUS]: 189,
  [GLFWKey.KEY_NUM_LOCK]: 144,
  [GLFWKey.KEY_PAGE_DOWN]: 34,
  [GLFWKey.KEY_PAGE_UP]: 33,
  [GLFWKey.KEY_PAUSE]: 19,
  [GLFWKey.KEY_PERIOD]: 190,
  [GLFWKey.KEY_PRINT_SCREEN]: 144,
  [GLFWKey.KEY_RIGHT]: 39,
  [GLFWKey.KEY_RIGHT_ALT]: 18,
  [GLFWKey.KEY_RIGHT_BRACKET]: 221,
  [GLFWKey.KEY_RIGHT_CONTROL]: 17,
  [GLFWKey.KEY_RIGHT_SHIFT]: 16,
  [GLFWKey.KEY_RIGHT_SUPER]: 93,
  [GLFWKey.KEY_SCROLL_LOCK]: 145,
  [GLFWKey.KEY_SEMICOLON]: 186,
  [GLFWKey.KEY_SLASH]: 191,
  [GLFWKey.KEY_SPACE]: 32,
  [GLFWKey.KEY_TAB]: 9,
  [GLFWKey.KEY_UP]: 38,
};
