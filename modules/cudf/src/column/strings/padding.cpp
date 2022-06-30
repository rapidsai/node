// Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <node_cudf/column.hpp>
#include <node_cudf/utilities/napi_to_cpp.hpp>

#include <node_rmm/memory_resource.hpp>

#include <cudf/strings/padding.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

namespace nv {

Column::wrapper_t Column::pad(cudf::size_type width,
                              cudf::strings::pad_side pad_side,
                              std::string const& fill_char,
                              rmm::mr::device_memory_resource* mr) const {
  try {
    return Column::New(Env(), cudf::strings::pad(this->view(), width, pad_side, fill_char, mr));
  } catch (std::exception const& e) { throw Napi::Error::New(Env(), e.what()); }
}

Column::wrapper_t Column::zfill(cudf::size_type width, rmm::mr::device_memory_resource* mr) const {
  try {
    return Column::New(Env(), cudf::strings::zfill(this->view(), width, mr));
  } catch (std::exception const& e) { throw Napi::Error::New(Env(), e.what()); }
}

Napi::Value Column::pad(Napi::CallbackInfo const& info) {
  if (info.Length() < 3) {
    NODE_CUDF_THROW("Column pad expects a width, pad_side, fill_char, and optional MemoryResource",
                    info.Env());
  }
  CallbackArgs const args{info};

  const std::string pad_size_string = args[1];
  const auto pad_side               = [&pad_size_string, &info]() {
    if (pad_size_string == "left") {
      return cudf::strings::pad_side::LEFT;
    } else if (pad_size_string == "right") {
      return cudf::strings::pad_side::RIGHT;
    } else if (pad_size_string == "both") {
      return cudf::strings::pad_side::BOTH;
    } else {
      NODE_CUDF_THROW("Invalid pad side " + pad_size_string, info.Env());
    }
  }();

  const std::string fill_char = args[2];
  if (fill_char.length() != 1) {
    NODE_CUDF_THROW("fill_char must be exactly one character", info.Env());
  }

  return pad(args[0], pad_side, fill_char, args[3]);
}

Napi::Value Column::zfill(Napi::CallbackInfo const& info) {
  if (info.Length() < 1) {
    NODE_CUDF_THROW("Column zfill expects a width and optional MemoryResource", info.Env());
  }
  CallbackArgs const args{info};
  return zfill(args[0], args[1]);
}

}  // namespace nv
