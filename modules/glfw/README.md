### node-glfw (`npm install @rapidsai/glfw`)
    A node native addon that provides bindings to the platform window manager via GLFW (https://www.glfw.org/).
    GLFW provides cross-platform multi-display support for creating native windows that host an
    OpenGL or Vulkan rendering context. These bindings provide a stripped-down version of the
    DOM's "Window", "Document", and "Canvas" APIs.

#### dependencies:
- `@rapidsai/webgl`

#### Window management:
- [Window](https://www.glfw.org/docs/latest/group__window.html) wraps and manages a GLFW `Window`.
- [Document](https://developer.mozilla.org/en-US/docs/Web/API/Document) implements a few DOM `Document` APIs that frameworks expect.
- [Canvas](https://developer.mozilla.org/en-US/docs/Web/API/Canvas_API) implements a DOM `Canvas` APIs that provides a 3D rendering context.

#### GLFW APIs:
    createWindow, showWindow, hideWindow, focusWindow, iconifyWindow, restoreWindow, maximizeWindow, requestWindowAttention, reparentWindow, destroyWindow, getWindowFrameSize, getWindowContentScale, setWindowSizeLimits, getPlatformWindowPointer, getPlatformContextPointer, get/setWindowIcon, get/setWindowSize, get/setWindowTitle, get/setWindowMonitor, get/setWindowOpacity, get/setWindowPosition, get/setWindowAttribute, get/setWindowAspectRatio, get/setWindowShouldClose
