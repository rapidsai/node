// Copyright (c) 2022 NVIDIA Corporation.

const reducer =
  (state, action) => {
    switch (action.type) {
      case 'MOUSE_MOVE':
        return { ...state, mouseX: action.event.screenX, mouseY: action.event.screenY, }
      case 'MOUSE_CLICK':
        console.log('click');
        console.log(state);
        return { ...state, isHeld: true }
      case 'MOUSE_RELEASE':
        console.log('unclick');
        return { ...state, isHeld: false }
      case 'SCROLL':
        console.log('scroll');
        const zoom   = action.event.deltaY > 0 ? -1 : 1;
        const result = action.event.deltaY > 0 ? state.zoomLevel / 1.2 : state.zoomLevel * 1.2;
        return { ...state, zoomLevel: result }
      case 'ROTATE':
        const angle = state.angle;
        return { ...state, angle: angle + 1 % 360 }
      case 'UPDATE_TRANSFORM':
        console.log('update transform...');
        console.log(action.event);
        return { ...state, mapTransform: action.event }
      default: throw new Error('chalupa batman');
    }
  }

export default reducer;
