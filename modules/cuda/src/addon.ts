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

/* eslint-disable @typescript-eslint/no-redeclare */

export const {
  VERSION,
  IPC_HANDLE_SIZE,
  getDriverVersion,
  rgbaMirror,
  bgraToYCrCb420,

  Math,
  driver,
  runtime,

  CUDAArray,
  ChannelFormatKind,

  Device,
  DeviceFlags,

  DeviceMemory,
  PinnedMemory,
  ManagedMemory,
  IpcMemory,
  IpcHandle,
  MappedGLMemory,
  _cpp_exports,
} = require('bindings')('rapidsai_cuda.node').init() as typeof import('./node_cuda');

export type getDriverVersion = typeof import('./node_cuda').getDriverVersion;
export type rgbaMirror       = typeof import('./node_cuda').rgbaMirror;
export type bgraToYCrCb420   = typeof import('./node_cuda').bgraToYCrCb420;

export type Math    = typeof import('./node_cuda').Math;
export type driver  = typeof import('./node_cuda').driver;
export type runtime = typeof import('./node_cuda').runtime;

export type Memory         = import('./node_cuda').Memory;
export type DeviceMemory   = import('./node_cuda').DeviceMemory;
export type PinnedMemory   = import('./node_cuda').PinnedMemory;
export type ManagedMemory  = import('./node_cuda').ManagedMemory;
export type IpcMemory      = import('./node_cuda').IpcMemory;
export type IpcHandle      = import('./node_cuda').IpcHandle;
export type MappedGLMemory = import('./node_cuda').MappedGLMemory;

export type CUDAArray         = import('./node_cuda').CUDAArray;
export type ChannelFormatKind = import('./node_cuda').ChannelFormatKind;

export type Device           = import('./node_cuda').Device;
export type DeviceFlags      = import('./node_cuda').DeviceFlags;
export type DeviceProperties = import('./node_cuda').DeviceProperties;
