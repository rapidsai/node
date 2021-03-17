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

#include <cudf/unary.hpp>

#include <rmm/mr/device/per_device_resource.hpp>
#include <cudf/strings/contains.hpp>

namespace nv {

ObjectUnwrap<Column> Column::contains_re(std::string const& pattern,
                                         rmm::mr::device_memory_resource* mr) const {
  try {
    return Column::New(cudf::strings::contains_re(this->view(), pattern, mr));
  } catch (cudf::logic_error const& err) { NAPI_THROW(Napi::Error::New(Env(), err.what())); }
}

ObjectUnwrap<Column> Column::count_re(std::string const& pattern,
                                      rmm::mr::device_memory_resource* mr) const {
  try {
    return Column::New(cudf::strings::count_re(this->view(), pattern, mr));
  } catch (cudf::logic_error const& err) { NAPI_THROW(Napi::Error::New(Env(), err.what())); }
}

ObjectUnwrap<Column> Column::matches_re(std::string const& pattern,
                                        rmm::mr::device_memory_resource* mr) const {
  try {
    return Column::New(cudf::strings::matches_re(this->view(), pattern, mr));
  } catch (cudf::logic_error const& err) { NAPI_THROW(Napi::Error::New(Env(), err.what())); }
}

Napi::Value Column::contains_re(Napi::CallbackInfo const& info) {
  if (info.Length() < 1) {
    NODE_CUDF_THROW("Column contains_re expects a pattern and optional MemoryResource", info.Env());
  }
  return contains_re(NapiToCPP{info[0]},
                     NapiToCPP(info[1]).operator rmm::mr::device_memory_resource*());
}

Napi::Value Column::count_re(Napi::CallbackInfo const& info) {
  if (info.Length() < 1) {
    NODE_CUDF_THROW("Column contains_re expects a pattern and optional MemoryResource", info.Env());
  }
  return count_re(NapiToCPP{info[0]},
                  NapiToCPP(info[1]).operator rmm::mr::device_memory_resource*());
}

Napi::Value Column::matches_re(Napi::CallbackInfo const& info) {
  if (info.Length() < 1) {
    NODE_CUDF_THROW("Column contains_re expects a pattern and optional MemoryResource", info.Env());
  }
  return matches_re(NapiToCPP{info[0]},
                    NapiToCPP(info[1]).operator rmm::mr::device_memory_resource*());
}

}  // namespace nv
