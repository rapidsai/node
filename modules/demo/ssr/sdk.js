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

const Rx = require('rxjs');
const { GLFWModifierKey, GLFWKey } = require('@nvidia/glfw');
const {
  KeyboardEvent,
  KeyCode,
  KeyState,
  MediaType,
  ModifierFlags,
  // MouseButton,
  MouseEvent,
  MouseEventType,
  Server,
  Stream,
  WindowEvent,
  WindowEventType,
} = require('node-nvidia-stream-sdk');

module.exports = { streamSDKServer, videoStream, inputStream, inputToDOMEvent };

function streamSDKServer(options = {}) {
  const onClientCommand = new Rx.Subject;
  const onClientConnecting = new Rx.Subject;
  const onClientConnected = new Rx.Subject;
  const onClientDisconnected = new Rx.Subject;
  const onDiagnosticsEvent = new Rx.Subject;
  const onSignalingHeaders = new Rx.Subject;
  const server = new Server({
    onClientCommand(command) { onClientCommand.next({ command }); },
    onClientConnecting(event) { onClientConnecting.next({ event }); },
    onClientConnected(event) { onClientConnected.next({ event }); },
    onClientDisconnected(event) { onClientDisconnected.next({ event }); },
    onDiagnosticsEvent(data) { onDiagnosticsEvent.next({ data }); },
    onSignalingHeaders(headers) { onSignalingHeaders.next({ headers }); },
    ...options,
  });
  return {
    server,
    onClientCommand,
    onClientConnecting,
    onClientConnected,
    onClientDisconnected,
    onDiagnosticsEvent,
    onSignalingHeaders,
  };
}

function videoStream(options = {}) {
  const onStreamUpdated = new Rx.Subject;
  const onStreamConnecting = new Rx.Subject;
  const onStreamConfigUpdateRequested = new Rx.Subject;
  const onStreamConnected = new Rx.Subject;
  const onStreamDisconnected = new Rx.Subject;
  const onQosStatusUpdated = new Rx.Subject;
  const onFrameInvalidationRequested = new Rx.Subject;
  const stream = new Stream({
    fps: 60,
    width: 1280,
    height: 720,
    enablePtd: true,
    ...options,
    mediaType: MediaType.VIDEO,
    onStreamUpdated(connection) { onStreamUpdated.next({ connection }); },
    onStreamConnecting(settings, config) { onStreamConnecting.next({ settings, config }); },
    onStreamConfigUpdateRequested(config) { onStreamConfigUpdateRequested.next({ config }); },
    onStreamConnected(connection) { onStreamConnected.next({ connection }); },
    onStreamDisconnected(connection) { onStreamDisconnected.next({ connection }); },
    onQosStatusUpdated(connection, qosStatus) { onQosStatusUpdated.next({ connection, qosStatus }); },
    onFrameInvalidationRequested(connection, invalidationRange) { onFrameInvalidationRequested.next({ connection, invalidationRange }); },
  });
  return {
    stream,
    onStreamUpdated,
    onStreamConnecting,
    onStreamConfigUpdateRequested,
    onStreamConnected,
    onStreamDisconnected,
    onQosStatusUpdated,
    onFrameInvalidationRequested,
  };
}

function inputStream(options = {}) {
  const onStreamUpdated = new Rx.Subject;
  const onStreamConnecting = new Rx.Subject;
  const onStreamConfigUpdateRequested = new Rx.Subject;
  const onStreamConnected = new Rx.Subject;
  const onStreamDisconnected = new Rx.Subject;
  const onQosStatusUpdated = new Rx.Subject;
  const onFrameInvalidationRequested = new Rx.Subject;
  const onClientInputReceived = new Rx.Subject;
  const stream = new Stream({
    ...options,
    mediaType: MediaType.INPUT,
    onStreamUpdated(connection) { onStreamUpdated.next({ connection }); },
    onStreamConnecting(settings, config) { onStreamConnecting.next({ settings, config }); },
    onStreamConfigUpdateRequested(config) { onStreamConfigUpdateRequested.next({ config }); },
    onStreamConnected(connection) { onStreamConnected.next({ connection }); },
    onStreamDisconnected(connection) { onStreamDisconnected.next({ connection }); },
    onQosStatusUpdated(connection, qosStatus) { onQosStatusUpdated.next({ connection, qosStatus }); },
    onFrameInvalidationRequested(connection, invalidationRange) { onFrameInvalidationRequested.next({ connection, invalidationRange }); },
    onClientInputReceived(connection, event) { onClientInputReceived.next({ connection, event }); },
  });
  return {
    stream,
    onStreamUpdated,
    onStreamConnecting,
    onStreamConfigUpdateRequested,
    onStreamConnected,
    onStreamDisconnected,
    onQosStatusUpdated,
    onFrameInvalidationRequested,
    onClientInputReceived,
  };
}

function inputToDOMEvent(window, event) {
  if (event instanceof MouseEvent) {
    return mouseToDOMEvent(window, event);
  } else if (event instanceof WindowEvent) {
    return windowToDOMEvent(window, event);
  } else if (event instanceof KeyboardEvent) {
    return keyboardToDOMEvent(window, event);
  }
}

function mouseToDOMEvent(window, { type, flags, xPos, yPos, data1, data2, vWheel, hWheel }) {

  const event = {
    modifiers:
      ((flags & ModifierFlags.ALT) || (flags & ModifierFlags.ALT_RIGHT) ? GLFWModifierKey.MOD_ALT : 0) |
      ((flags & ModifierFlags.META) || (flags & ModifierFlags.META_RIGHT) ? GLFWModifierKey.MOD_SUPER : 0) |
      ((flags & ModifierFlags.SHIFT) || (flags & ModifierFlags.SHIFT_RIGHT) ? GLFWModifierKey.MOD_SHIFT : 0) |
      ((flags & ModifierFlags.CONTROL) || (flags & ModifierFlags.CONTROL_RIGHT) ? GLFWModifierKey.MOD_CONTROL : 0) |
      (window.capsLock ? GLFWModifierKey.MOD_CAPS_LOCK : 0)
  };

  switch (type) {
    case MouseEventType.NONE: break;
    case MouseEventType.BUTTON:
      if (data2 === KeyState.NONE) break;
      // if (data1 === /* MouseButton.LEFT */ 1) event.button = 0;
      // if (data1 === /* MouseButton.MIDDLE */ 2) event.button = 1;
      // if (data1 === /* MouseButton.RIGHT */ 3) event.button = 2;
      // ^ is the same as `data1 - 1`
      event.button = data1 - 1;
      event.x = xPos;
      event.y = yPos;
      if (!(flags & ModifierFlags.ABSCOORDS)) {
        event.x += window.mouseX;
        event.y += window.mouseY;
      }
      event.type = data2 === KeyState.DOWN ? 'mousedown' : 'mouseup';
      return event;
    case MouseEventType.MOUSE_MOVE:
      event.type = 'mousemove';
      event.x = xPos;
      event.y = yPos;
      if (!(flags & ModifierFlags.ABSCOORDS)) {
        event.x += window.mouseX;
        event.y += window.mouseY;
      }
      return event;
    case MouseEventType.MOUSE_WHEEL:
      event.type = 'wheel';
      event.deltaY = -vWheel / 10;
      event.deltaX = -hWheel / 10;
      return event;
    default: break;
  }
}

function windowToDOMEvent(window, { type, windowRect }) {
  const event = {};
  switch (type) {
    case WindowEventType.NONE: return;
    case WindowEventType.SET_POS_MOVE:
      event.type = 'move';
      event.x = windowRect.x1;
      event.y = windowRect.y1;
      break;
    case WindowEventType.SET_POS_RESIZE:
      event.type = 'resize';
      event.width = windowRect.x2 - windowRect.x1;
      event.height = windowRect.y2 - windowRect.y1;
      break;
    case WindowEventType.MINIMIZE:
      event.type = 'minimize';
      break;
    case WindowEventType.MAXIMIZE:
      event.type = 'maximize';
      break;
    case WindowEventType.RESTORE:
      event.type = 'restore';
      break;
    case WindowEventType.CLOSE:
      event.type = 'close';
      break;
    case WindowEventType.GAIN_FOCUS:
      event.type = 'focus';
      break;
    case WindowEventType.LOSE_FOCUS:
      event.type = 'blur';
      break;
    case WindowEventType.ENTER_WINDOW:
      event.type = 'mouseenter';
      event.x = window.mouseX;
      event.y = window.mouseY;
      break;
    case WindowEventType.LEAVE_WINDOW:
      event.type = 'mouseleave';
      event.x = window.mouseX;
      event.y = window.mouseY;
      break;
    default: return;
  }
  return event;
}

const streamSDKKeyToGLFWKey = (() => {
  const keyCodes = {};
  for (const sdkName in KeyCode) {
    const glfwName = `KEY_${sdkName}`;
    if (GLFWKey[glfwName]) {
      keyCodes[KeyCode[sdkName]] = GLFWKey[glfwName];
    }
  }
  return keyCodes;
})();

function keyboardToDOMEvent(window, { keyCode, scanCode, flags, keyState }) {
  if (keyState === KeyState.NONE) return;
  const key = streamSDKKeyToGLFWKey[keyCode];
  if (key === undefined) return;
  const altKey = (flags & ModifierFlags.ALT) || (flags & ModifierFlags.ALT_RIGHT) || [KeyCode.ALT, KeyCode.LALT, KeyCode.RALT].some((x) => keyCode === x);
  const superKey = (flags & ModifierFlags.META) || (flags & ModifierFlags.META_RIGHT) || [KeyCode.META, KeyCode.LMETA, KeyCode.RMETA].some((x) => keyCode === x);
  const shiftKey = (flags & ModifierFlags.SHIFT) || (flags & ModifierFlags.SHIFT_RIGHT) || [KeyCode.SHIFT, KeyCode.LSHIFT, KeyCode.RSHIFT].some((x) => keyCode === x);
  const controlKey = (flags & ModifierFlags.CONTROL) || (flags & ModifierFlags.CONTROL_RIGHT) || [KeyCode.CONTROL, KeyCode.LCONTROL, KeyCode.RCONTROL].some((x) => keyCode === x);
  const capsLock = (keyCode === KeyCode.CAPS_LOCK) || window.capsLock;
  return {
    key,
    scancode: scanCode,
    type: keyState === KeyState.DOWN ? 'keydown' : 'keyup',
    modifiers:
      (altKey ? GLFWModifierKey.MOD_ALT : 0) |
      (superKey ? GLFWModifierKey.MOD_SUPER : 0) |
      (shiftKey ? GLFWModifierKey.MOD_SHIFT : 0) |
      (controlKey ? GLFWModifierKey.MOD_CONTROL : 0) |
      (capsLock ? GLFWModifierKey.MOD_CAPS_LOCK : 0)
  };
}
