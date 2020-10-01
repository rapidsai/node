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

#include "math.hpp"
#include "cuda/utilities/napi_to_cpp.hpp"
#include "macros.hpp"

#include <nv_node/utilities/args.hpp>

namespace nv {

Napi::Value math_abs(CallbackArgs const &info) {
  return nv::math::dispatch(nv::math::calc_abs{}, info);
}
Napi::Value math_acos(CallbackArgs const &info) {
  return nv::math::dispatch(nv::math::calc_acos{}, info);
}
Napi::Value math_asin(CallbackArgs const &info) {
  return nv::math::dispatch(nv::math::calc_asin{}, info);
}
Napi::Value math_atan(CallbackArgs const &info) {
  return nv::math::dispatch(nv::math::calc_atan{}, info);
}
Napi::Value math_atan2(CallbackArgs const &info) {
  return nv::math::dispatch(nv::math::calc_atan2{}, info);
}
Napi::Value math_ceil(CallbackArgs const &info) {
  return nv::math::dispatch(nv::math::calc_ceil{}, info);
}
Napi::Value math_cos(CallbackArgs const &info) {
  return nv::math::dispatch(nv::math::calc_cos{}, info);
}
Napi::Value math_exp(CallbackArgs const &info) {
  return nv::math::dispatch(nv::math::calc_exp{}, info);
}
Napi::Value math_floor(CallbackArgs const &info) {
  return nv::math::dispatch(nv::math::calc_floor{}, info);
}
Napi::Value math_log(CallbackArgs const &info) {
  return nv::math::dispatch(nv::math::calc_log{}, info);
}
Napi::Value math_max(CallbackArgs const &info) {
  return nv::math::dispatch(nv::math::calc_max{}, info);
}
Napi::Value math_min(CallbackArgs const &info) {
  return nv::math::dispatch(nv::math::calc_min{}, info);
}
Napi::Value math_pow(CallbackArgs const &info) {
  return nv::math::dispatch(nv::math::calc_pow{}, info);
}
Napi::Value math_round(CallbackArgs const &info) {
  return nv::math::dispatch(nv::math::calc_round{}, info);
}
Napi::Value math_sin(CallbackArgs const &info) {
  return nv::math::dispatch(nv::math::calc_sin{}, info);
}
Napi::Value math_sqrt(CallbackArgs const &info) {
  return nv::math::dispatch(nv::math::calc_sqrt{}, info);
}
Napi::Value math_tan(CallbackArgs const &info) {
  return nv::math::dispatch(nv::math::calc_tan{}, info);
}

namespace math {
Napi::Object initModule(Napi::Env env, Napi::Object exports) {
  EXPORT_FUNC(env, exports, "abs", math_abs);
  EXPORT_FUNC(env, exports, "acos", math_acos);
  EXPORT_FUNC(env, exports, "asin", math_asin);
  EXPORT_FUNC(env, exports, "atan", math_atan);
  EXPORT_FUNC(env, exports, "atan2", math_atan2);
  EXPORT_FUNC(env, exports, "ceil", math_ceil);
  EXPORT_FUNC(env, exports, "cos", math_cos);
  EXPORT_FUNC(env, exports, "exp", math_exp);
  EXPORT_FUNC(env, exports, "floor", math_floor);
  EXPORT_FUNC(env, exports, "log", math_log);
  EXPORT_FUNC(env, exports, "max", math_max);
  EXPORT_FUNC(env, exports, "min", math_min);
  EXPORT_FUNC(env, exports, "pow", math_pow);
  EXPORT_FUNC(env, exports, "round", math_round);
  EXPORT_FUNC(env, exports, "sin", math_sin);
  EXPORT_FUNC(env, exports, "sqrt", math_sqrt);
  EXPORT_FUNC(env, exports, "tan", math_tan);
  return exports;
}
}  // namespace math
}  // namespace nv
