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

#include <cudf/strings/attributes.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

namespace nv {

Column::wrapper_t Column::count_characters(rmm::mr::device_memory_resource* mr) const {
  try {
    return Column::New(Env(), cudf::strings::count_characters(this->view(), mr));
  } catch (std::exception const& e) { throw Napi::Error::New(Env(), e.what()); }
}

Column::wrapper_t Column::count_bytes(rmm::mr::device_memory_resource* mr) const {
  try {
    return Column::New(Env(), cudf::strings::count_bytes(this->view(), mr));
  } catch (std::exception const& e) { throw Napi::Error::New(Env(), e.what()); }
}

Napi::Value Column::count_characters(Napi::CallbackInfo const& info) {
  return count_characters(NapiToCPP(info[0]).operator rmm::mr::device_memory_resource*());
}

Napi::Value Column::count_bytes(Napi::CallbackInfo const& info) {
  return count_bytes(NapiToCPP(info[0]).operator rmm::mr::device_memory_resource*());
}

}  // namespace nv
