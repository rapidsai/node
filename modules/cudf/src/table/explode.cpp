// Copyright (c) 2022, NVIDIA CORPORATION.
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
#include <node_cudf/table.hpp>

#include <cudf/lists/explode.hpp>

namespace nv {

Table::wrapper_t Table::explode(cudf::size_type explode_column_idx,
                                rmm::mr::device_memory_resource* mr) const {
  try {
    return Table::New(Env(), cudf::explode(*this, explode_column_idx, mr));
  } catch (cudf::logic_error const& e) { NAPI_THROW(Napi::Error::New(Env(), e.what())); }
}

Napi::Value Table::explode(Napi::CallbackInfo const& info) {
  return explode(info[0].ToNumber(),
                 NapiToCPP(info[1]).operator rmm::mr::device_memory_resource*());
}

Table::wrapper_t Table::explode_position(cudf::size_type explode_column_idx,
                                         rmm::mr::device_memory_resource* mr) const {
  try {
    return Table::New(Env(), cudf::explode_position(*this, explode_column_idx, mr));
  } catch (cudf::logic_error const& e) { NAPI_THROW(Napi::Error::New(Env(), e.what())); }
}

Napi::Value Table::explode_position(Napi::CallbackInfo const& info) {
  return explode_position(info[0].ToNumber(),
                          NapiToCPP(info[1]).operator rmm::mr::device_memory_resource*());
}

Table::wrapper_t Table::explode_outer(cudf::size_type explode_column_idx,
                                      rmm::mr::device_memory_resource* mr) const {
  try {
    return Table::New(Env(), cudf::explode_outer(*this, explode_column_idx, mr));
  } catch (cudf::logic_error const& e) { NAPI_THROW(Napi::Error::New(Env(), e.what())); }
}

Napi::Value Table::explode_outer(Napi::CallbackInfo const& info) {
  return explode_outer(info[0].ToNumber(),
                       NapiToCPP(info[1]).operator rmm::mr::device_memory_resource*());
}

Table::wrapper_t Table::explode_outer_position(cudf::size_type explode_column_idx,
                                               rmm::mr::device_memory_resource* mr) const {
  try {
    return Table::New(Env(), cudf::explode_outer_position(*this, explode_column_idx, mr));
  } catch (cudf::logic_error const& e) { NAPI_THROW(Napi::Error::New(Env(), e.what())); }
}

Napi::Value Table::explode_outer_position(Napi::CallbackInfo const& info) {
  return explode_outer_position(info[0].ToNumber(),
                                NapiToCPP(info[1]).operator rmm::mr::device_memory_resource*());
}

}  // namespace nv
