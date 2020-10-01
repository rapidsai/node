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

#include <cmath>
#include <cstdint>
#include <random>

#include <nv_node/utilities/args.hpp>
#include "cuda/utilities/cpp_to_napi.hpp"
#include "cuda/utilities/napi_to_cpp.hpp"

namespace nv {
namespace math {

template <typename F>
inline Napi::Value dispatch(F f, CallbackArgs const& info) {
  return info[0].val.IsBigInt() ? CPPToNapi(info)(f.template operator()<int64_t>(info))
                                : CPPToNapi(info)(f.template operator()<double>(info));
}

struct calc_abs {
  template <typename T>
  inline auto operator()(CallbackArgs const& info) {
    return std::abs(info[0].operator T());
  }
};

struct calc_acos {
  template <typename T>
  inline auto operator()(CallbackArgs const& info) {
    return std::acos(info[0].operator T());
  }
};

struct calc_asin {
  template <typename T>
  inline auto operator()(CallbackArgs const& info) {
    return std::asin(info[0].operator T());
  }
};

struct calc_atan {
  template <typename T>
  inline auto operator()(CallbackArgs const& info) {
    return std::atan(info[0].operator T());
  }
};

struct calc_atan2 {
  template <typename T>
  inline auto operator()(CallbackArgs const& info) {
    return std::atan2(info[0].operator T(), info[1].operator T());
  }
};

struct calc_ceil {
  template <typename T>
  inline auto operator()(CallbackArgs const& info) {
    return std::ceil(info[0].operator T());
  }
};

struct calc_cos {
  template <typename T>
  inline auto operator()(CallbackArgs const& info) {
    return std::cos(info[0].operator T());
  }
};

struct calc_exp {
  template <typename T>
  inline auto operator()(CallbackArgs const& info) {
    return std::exp(info[0].operator T());
  }
};

struct calc_floor {
  template <typename T>
  inline auto operator()(CallbackArgs const& info) {
    return std::floor(info[0].operator T());
  }
};

struct calc_log {
  template <typename T>
  inline auto operator()(CallbackArgs const& info) {
    return std::log(info[0].operator T());
  }
};

struct calc_max {
  template <typename T>
  inline auto operator()(CallbackArgs const& info) {
    return std::max(info[0].operator T(), info[1].operator T());
  }
};

struct calc_min {
  template <typename T>
  inline auto operator()(CallbackArgs const& info) {
    return std::min(info[0].operator T(), info[1].operator T());
  }
};

struct calc_pow {
  template <typename T>
  inline auto operator()(CallbackArgs const& info) {
    return std::pow(info[0].operator T(), info[1].operator T());
  }
};

struct calc_round {
  template <typename T>
  inline auto operator()(CallbackArgs const& info) {
    return std::round(info[0].operator T());
  }
};

struct calc_sin {
  template <typename T>
  inline auto operator()(CallbackArgs const& info) {
    return std::sin(info[0].operator T());
  }
};

struct calc_sqrt {
  template <typename T>
  inline auto operator()(CallbackArgs const& info) {
    return std::sqrt(info[0].operator T());
  }
};

struct calc_tan {
  template <typename T>
  inline auto operator()(CallbackArgs const& info) {
    return std::tan(info[0].operator T());
  }
};

}  // namespace math
}  // namespace nv
