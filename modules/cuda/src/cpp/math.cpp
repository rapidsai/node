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

#include <node_cuda/macros.hpp>
#include <node_cuda/math.hpp>

namespace {

using FromJS = node_cuda::FromJS;
using ToNapi = node_cuda::ToNapi;

Napi::Value math_abs(Napi::CallbackInfo const &info) {
  return node_cuda::math::dispatch(node_cuda::math::calc_abs{}, info);
}
Napi::Value math_acos(Napi::CallbackInfo const &info) {
  return node_cuda::math::dispatch(node_cuda::math::calc_acos{}, info);
}
Napi::Value math_asin(Napi::CallbackInfo const &info) {
  return node_cuda::math::dispatch(node_cuda::math::calc_asin{}, info);
}
Napi::Value math_atan(Napi::CallbackInfo const &info) {
  return node_cuda::math::dispatch(node_cuda::math::calc_atan{}, info);
}
Napi::Value math_atan2(Napi::CallbackInfo const &info) {
  return node_cuda::math::dispatch(node_cuda::math::calc_atan2{}, info);
}
Napi::Value math_ceil(Napi::CallbackInfo const &info) {
  return node_cuda::math::dispatch(node_cuda::math::calc_ceil{}, info);
}
Napi::Value math_cos(Napi::CallbackInfo const &info) {
  return node_cuda::math::dispatch(node_cuda::math::calc_cos{}, info);
}
Napi::Value math_exp(Napi::CallbackInfo const &info) {
  return node_cuda::math::dispatch(node_cuda::math::calc_exp{}, info);
}
Napi::Value math_floor(Napi::CallbackInfo const &info) {
  return node_cuda::math::dispatch(node_cuda::math::calc_floor{}, info);
}
Napi::Value math_log(Napi::CallbackInfo const &info) {
  return node_cuda::math::dispatch(node_cuda::math::calc_log{}, info);
}
Napi::Value math_max(Napi::CallbackInfo const &info) {
  return node_cuda::math::dispatch(node_cuda::math::calc_max{}, info);
}
Napi::Value math_min(Napi::CallbackInfo const &info) {
  return node_cuda::math::dispatch(node_cuda::math::calc_min{}, info);
}
Napi::Value math_pow(Napi::CallbackInfo const &info) {
  return node_cuda::math::dispatch(node_cuda::math::calc_pow{}, info);
}
Napi::Value math_round(Napi::CallbackInfo const &info) {
  return node_cuda::math::dispatch(node_cuda::math::calc_round{}, info);
}
Napi::Value math_sin(Napi::CallbackInfo const &info) {
  return node_cuda::math::dispatch(node_cuda::math::calc_sin{}, info);
}
Napi::Value math_sqrt(Napi::CallbackInfo const &info) {
  return node_cuda::math::dispatch(node_cuda::math::calc_sqrt{}, info);
}
Napi::Value math_tan(Napi::CallbackInfo const &info) {
  return node_cuda::math::dispatch(node_cuda::math::calc_tan{}, info);
}
}  // namespace

namespace node_cuda {

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
}  // namespace node_cuda
