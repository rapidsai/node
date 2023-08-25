// Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

#include "cpp_to_napi.hpp"
#include "napi_to_cpp.hpp"

#include <napi.h>

namespace nv {

struct CallbackArgs {
  // Constructor that accepts the same arguments as the Napi::CallbackInfo constructor
  CallbackArgs(napi_env env, napi_callback_info info)
    : CallbackArgs(new Napi::CallbackInfo(env, info), true) {}

  // Construct a CallbackArgs by proxying to an Napi::CallbackInfo instance
  CallbackArgs(Napi::CallbackInfo const* info, bool owns_info = false)
    : owns_info_(owns_info), info_(info){};

  // Construct a CallbackArgs by proxying to an Napi::CallbackInfo instance
  CallbackArgs(Napi::CallbackInfo const& info) : info_(&info){};

  ~CallbackArgs() {
    if (owns_info_) { delete info_; }
    info_ = nullptr;
  }

  // Proxy all the public methods
  Napi::Env Env() const { return info_->Env(); }
  Napi::Value NewTarget() const { return info_->NewTarget(); }
  bool IsConstructCall() const { return info_->IsConstructCall(); }
  size_t Length() const { return info_->Length(); }
  Napi::Value This() const { return info_->This(); }
  void* Data() const { return info_->Data(); }
  void SetData(void* data) { const_cast<Napi::CallbackInfo*>(info_)->SetData(data); }

  // the [] operator returns instances with implicit conversion operators from JS to C++ types
  NapiToCPP const operator[](size_t i) const { return info_->operator[](i); }

  inline Napi::CallbackInfo const& info() const { return *info_; }
  inline operator Napi::CallbackInfo const&() const { return *info_; }

 private:
  bool owns_info_{false};
  Napi::CallbackInfo const* info_{nullptr};
};

}  // namespace nv
