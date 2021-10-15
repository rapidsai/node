### node-cuda (`npm install @rapidsai/cuda`)
    A node native addon that provides bindings to the CUDA driver and runtime APIs.
    These bindings allow calling the CUDA device management, memory, stream, event, ipc, and
    DX/GL interop APIs from JS. These APIs allow node (v8 or chakracore) applications to read,
    write, and share memory via zero-copy CUDA IPC with external processes that also use the
    CUDA, OpenGL, and RAPIDS libraries.

#### Device management:
    cudaChooseDevice, cudaGetDeviceCount, cuDeviceGet, cudaDeviceGetPCIBusId, cudaDeviceGetByPCIBusId, cudaGetDevice, cudaGetDeviceFlags, cudaGetDeviceProperties, cudaSetDevice, cudaSetDeviceFlags, cudaDeviceReset, cudaDeviceSynchronize, cudaDeviceCanAccessPeer, cudaDeviceEnablePeerAccess, cudaDeviceDisablePeerAccess

#### Memory:
- `CUDADevice`: A class to wrap and manage a CUDA device.
- `CUDABuffer`: A class to wrap and manage device memory allocations (similar to ArrayBuffer).
- `CUDAArray`: A class to wrap operations like read/write/share on a `CUDABuffer` (similar to TypedArray).
-     cuPointerGetAttribute, cudaMalloc, cudaFree, cudaMallocHost, cudaFreeHost, cudaHostRegister, cudaHostUnregister, cudaMemcpy, cudaMemset, cudaMemcpyAsync, cudaMemsetAsync, cudaMemGetInfo

#### IPC:
    cudaIpcGetMemHandle, cudaIpcOpenMemHandle, cudaIpcCloseMemHandle

#### Stream:
    cudaStreamCreate, cudaStreamDestroy, cudaStreamSynchronize

#### OpenGL:
    cuGraphicsGLRegisterBuffer, cuGraphicsGLRegisterImage, cuGraphicsUnregisterResource, cuGraphicsMapResources, cuGraphicsUnapResources, cuGraphicsResourceGetMappedPointer
