// Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

#include <cudf/reshape.hpp>

namespace nv {

Column::wrapper_t Table::interleave_columns(rmm::mr::device_memory_resource* mr) const {
  try {
    return Column::New(Env(), cudf::interleave_columns(*this, mr));
  } catch (std::exception const& e) { throw Napi::Error::New(Env(), e.what()); }
}

Napi::Value Table::interleave_columns(Napi::CallbackInfo const& info) {
  return interleave_columns(NapiToCPP(info[0]).operator rmm::mr::device_memory_resource*());
}

}  // namespace nv
