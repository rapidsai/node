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

#include <cudf/concatenate.hpp>
#include <node_cudf/column.hpp>

namespace nv {

Column::wrapper_t Column::concat(cudf::column_view const& other,
                                 rmm::mr::device_memory_resource* mr) {
  try {
    return Column::New(Env(), cudf::concatenate(std::vector{this->view(), other}, mr));
  } catch (std::exception const& e) { NAPI_THROW(Napi::Error::New(Env(), e.what())); }
}

Napi::Value Column::concat(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};
  return concat(*Column::Unwrap(args[0]), args[1]);
}

}  // namespace nv
