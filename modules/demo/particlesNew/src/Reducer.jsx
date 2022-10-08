/**
 * Copyright (c) Mik BRY
 * mik@miklabs.com
 *
 * This source code is licensed under private license found in the
 * LICENSE file in the root directory of this source tree.
 */

const reducer = (state, action) => {
  switch (action.type) {
    case 'MOUSE_MOVE':
      if (state.isHeld) {
        return {
          ...state,
          centerX: state.centerX + action.event.screenX - state.mouseX,
          centerY: state.centerY + action.event.screenY - state.mouseY,
          mouseX: action.event.screenX,
          mouseY: action.event.screenY,
        }
      }
      else { return state; }
    case 'MOUSE_CLICK':
      console.log('click');
      return {
        ...state,
        isHeld: true,
        mouseX: action.event.screenX,
        mouseY: action.event.screenY,
      }
    case 'MOUSE_RELEASE':
      console.log('unclick');
      return {
        ...state,
        isHeld: false
      }
    case 'SCROLL':
      const zoom = action.event.deltaY > 0 ? 1 : -1;
      const result = state.zoomLevel + zoom;
      return {
        ...state,
        zoomLevel: result
      }
    case 'ROTATE':
      const angle = state.angle;
      console.log('angle event');
      return {
        ...state,
        angle: angle + 1 % 360
      }
    default:
      throw new Error('Unknown state event');
  }
}

export default reducer
