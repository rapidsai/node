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

#include <node_cuda/casting.hpp>

#include <cmath>
#include <cstdint>
#include <random>

namespace node_cuda {
namespace math {

using FromJS = node_cuda::FromJS;
using ToNapi = node_cuda::ToNapi;

template <typename F>
inline Napi::Value dispatch(F f, Napi::CallbackInfo const& info) {
  return info[0].IsBigInt() ? ToNapi(info.Env())(f.template operator()<int64_t>(info))
                            : ToNapi(info.Env())(f.template operator()<double>(info));
}

struct calc_abs {
  template <typename T>
  inline auto operator()(Napi::CallbackInfo const& info) {
    return std::abs(FromJS(info[0]).operator T());
  }
};

struct calc_acos {
  template <typename T>
  inline auto operator()(Napi::CallbackInfo const& info) {
    return std::acos(FromJS(info[0]).operator T());
  }
};

struct calc_asin {
  template <typename T>
  inline auto operator()(Napi::CallbackInfo const& info) {
    return std::asin(FromJS(info[0]).operator T());
  }
};

struct calc_atan {
  template <typename T>
  inline auto operator()(Napi::CallbackInfo const& info) {
    return std::atan(FromJS(info[0]).operator T());
  }
};

struct calc_atan2 {
  template <typename T>
  inline auto operator()(Napi::CallbackInfo const& info) {
    return std::atan2(FromJS(info[0]).operator T(), FromJS(info[1]).operator T());
  }
};

struct calc_ceil {
  template <typename T>
  inline auto operator()(Napi::CallbackInfo const& info) {
    return std::ceil(FromJS(info[0]).operator T());
  }
};

struct calc_cos {
  template <typename T>
  inline auto operator()(Napi::CallbackInfo const& info) {
    return std::cos(FromJS(info[0]).operator T());
  }
};

struct calc_exp {
  template <typename T>
  inline auto operator()(Napi::CallbackInfo const& info) {
    return std::exp(FromJS(info[0]).operator T());
  }
};

struct calc_floor {
  template <typename T>
  inline auto operator()(Napi::CallbackInfo const& info) {
    return std::floor(FromJS(info[0]).operator T());
  }
};

struct calc_log {
  template <typename T>
  inline auto operator()(Napi::CallbackInfo const& info) {
    return std::log(FromJS(info[0]).operator T());
  }
};

struct calc_max {
  template <typename T>
  inline auto operator()(Napi::CallbackInfo const& info) {
    return std::max(FromJS(info[0]).operator T(), FromJS(info[1]).operator T());
  }
};

struct calc_min {
  template <typename T>
  inline auto operator()(Napi::CallbackInfo const& info) {
    return std::min(FromJS(info[0]).operator T(), FromJS(info[1]).operator T());
  }
};

struct calc_pow {
  template <typename T>
  inline auto operator()(Napi::CallbackInfo const& info) {
    return std::pow(FromJS(info[0]).operator T(), FromJS(info[1]).operator T());
  }
};

struct calc_round {
  template <typename T>
  inline auto operator()(Napi::CallbackInfo const& info) {
    return std::round(FromJS(info[0]).operator T());
  }
};

struct calc_sin {
  template <typename T>
  inline auto operator()(Napi::CallbackInfo const& info) {
    return std::sin(FromJS(info[0]).operator T());
  }
};

struct calc_sqrt {
  template <typename T>
  inline auto operator()(Napi::CallbackInfo const& info) {
    return std::sqrt(FromJS(info[0]).operator T());
  }
};

struct calc_tan {
  template <typename T>
  inline auto operator()(Napi::CallbackInfo const& info) {
    return std::tan(FromJS(info[0]).operator T());
  }
};

}  // namespace math
}  // namespace node_cuda