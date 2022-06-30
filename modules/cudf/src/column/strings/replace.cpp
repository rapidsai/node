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

#include <cudf/strings/replace.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

namespace nv {

Column::wrapper_t Column::replace_slice(std::string const& repl,
                                        cudf::size_type start,
                                        cudf::size_type stop,
                                        rmm::mr::device_memory_resource* mr) const {
  try {
    return Column::New(Env(), cudf::strings::replace_slice(this->view(), repl, start, stop, mr));
  } catch (std::exception const& e) { throw Napi::Error::New(Env(), e.what()); }
}

Napi::Value Column::replace_slice(Napi::CallbackInfo const& info) {
  if (info.Length() < 3) {
    NODE_CUDF_THROW(
      "Column replace_slice expects a replacement, start, stop, and optional MemoryResource",
      info.Env());
  }
  CallbackArgs const args{info};

  return replace_slice(args[0], args[1], args[2], args[3]);
}

}  // namespace nv
