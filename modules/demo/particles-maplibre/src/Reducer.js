// Copyright (c) 2022 NVIDIA Corporation.

const reducer =
  (state, action) => {
    switch (action.type) {
      case 'MOUSE_MOVE':
        return { ...state, mouseX: action.event.screenX, mouseY: action.event.screenY, }
      case 'MOUSE_CLICK':
        console.log('click');
        return { ...state, isHeld: true }
      case 'MOUSE_RELEASE':
        console.log('unclick');
        return { ...state, isHeld: false }
      case 'SCROLL':
        console.log('scroll');
        const zoom   = action.event.deltaY > 0 ? 1 : -1;
        const result = state.zoomLevel + zoom;
        return { ...state, zoomLevel: result }
      case 'ROTATE':
        const angle = state.angle;
        return { ...state, angle: angle + 1 % 360 }
      default: throw new Error('chalupa batman');
    }
  }

export default reducer;
