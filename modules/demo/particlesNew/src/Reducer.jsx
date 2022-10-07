/**
 * Copyright (c) Mik BRY
 * mik@miklabs.com
 *
 * This source code is licensed under private license found in the
 * LICENSE file in the root directory of this source tree.
 */

const toggle = previous => !previous;
const reducer = (state, action) => {
  switch (action.type) {
    case 'MOUSE_MOVE':
      return {
        ...state,
        mouseX: action.event.screenX,
        mouseY: action.event.screenY,
      }
    case 'MOUSE_CLICK':
      console.log('click');
      return {
        ...state,
        isHeld: toggle()
      }
    case 'MOUSE_RELEASE':
      console.log('unclick');
      return {
        ...state,
        isHeld: toggle()
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
      return {
        ...state,
        angle: angle + 1 % 360
      }
    default:
      throw new Error('Unknown state event');
  }
}

export default reducer
