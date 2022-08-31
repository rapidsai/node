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
#include <node_cudf/scalar.hpp>

#include <cudf/strings/json.hpp>

namespace nv {

Column::wrapper_t Column::get_json_object(std::string const& json_path,
                                          rmm::mr::device_memory_resource* mr) {
  auto options = cudf::strings::get_json_object_options{};
  options.set_missing_fields_as_nulls(true);
  try {
    auto col =
      Column::New(Env(),
                  cudf::strings::get_json_object(
                    this->view(), json_path, cudf::strings::get_json_object_options{}, mr));
    cudf::scalar& valid_count = *col->is_valid(mr)->sum(mr);
    auto& count_scalar        = static_cast<cudf::numeric_scalar<cudf::size_type>&>(valid_count);
    col->set_null_count(count_scalar.value());
    return col;
  } catch (std::exception const& e) { NAPI_THROW(Napi::Error::New(Env(), e.what())); }
}

Napi::Value Column::get_json_object(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};
  return get_json_object(args[0], args[1]);
}

}  // namespace nv
