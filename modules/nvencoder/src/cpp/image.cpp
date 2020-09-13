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

#include <node_nvencoder/casting.hpp>
#include <node_nvencoder/macros.hpp>

#include <napi.h>
#include <nppi.h>

namespace node_nvencoder {

// NppStatus nppiMirror_8u_C4R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep,
// NppiSize oROI, NppiAxis flip);
Napi::Value RGBAMirror(Napi::CallbackInfo const& info) {
  int32_t width                   = FromJS(info[0]);
  int32_t height                  = FromJS(info[1]);
  NppiAxis flip                   = static_cast<NppiAxis>(FromJS(info[2]).operator uint32_t());
  std::pair<size_t, uint8_t*> src = FromJS(info[3]);
  NppiSize roi                    = {width, height};
  if (info.Length() == 4) {
    nppiMirror_8u_C4IR(src.second, width * 4, roi, flip);
  } else if (info.Length() == 5) {
    std::pair<size_t, uint8_t*> dst = FromJS(info[4]);
    nppiMirror_8u_C4R(src.second, width * 4, dst.second, width * 4, roi, flip);
  }

  return info.Env().Undefined();
}

// NppStatus nppiBGRToYCrCb420_8u_AC4P3R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst[3], int
// rDstStep[3], NppiSize oSizeROI);
Napi::Value BGRAToYCrCb420(Napi::CallbackInfo const& info) {
  std::pair<size_t, uint8_t*> dst = FromJS(info[0]);
  std::pair<size_t, uint8_t*> src = FromJS(info[1]);
  int32_t width                   = FromJS(info[2]);
  int32_t height                  = FromJS(info[3]);
  NppiSize roi                    = {width, height};
  Npp8u* dstBuff[3]               = {
    dst.second, dst.second + width * height, dst.second + width * height * 5 / 4};

  int dstSteps[3] = {width, width / 2, width / 2};

  nppiBGRToYCrCb420_8u_AC4P3R(src.second, width * 4, dstBuff, dstSteps, roi);

  return info.Env().Undefined();
}

namespace image {
Napi::Value initModule(Napi::Env env, Napi::Object exports) {
  EXPORT_FUNC(env, exports, "rgbaMirror", node_nvencoder::RGBAMirror);
  EXPORT_FUNC(env, exports, "bgraToYCrCb420", node_nvencoder::BGRAToYCrCb420);
  return exports;
}
}  // namespace image

}  // namespace node_nvencoder
