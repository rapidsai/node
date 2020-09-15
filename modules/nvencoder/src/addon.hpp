// Copyright (c) 2020, NVIDIA CORPORATION.
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

#pragma once

#include <napi.h>
#include <nppi.h>
#include <nvEncodeAPI.h>

namespace nv {

Napi::Value nvencoder_init(Napi::CallbackInfo const& info);

namespace image {
Napi::Value initModule(Napi::Env env, Napi::Object exports);
}

// NppStatus nppiMirror_8u_C4R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep,
// NppiSize oROI, NppiAxis flip);
Napi::Value RGBAMirror(Napi::CallbackInfo const& info);

// NppStatus nppiBGRToYCrCb420_8u_AC4P3R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst[3], int
// rDstStep[3], NppiSize oSizeROI);
Napi::Value BGRAToYCrCb420(Napi::CallbackInfo const& info);

}  // namespace nv
