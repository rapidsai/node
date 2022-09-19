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

#include <node_cudf/utilities/napi_to_cpp.hpp>

#include <node_cudf/column.hpp>
#include <node_cudf/scalar.hpp>

#include <cudf/detail/null_mask.hpp>
#include <cudf/strings/json.hpp>

namespace nv {

Column::wrapper_t Column::get_json_object(std::string const& json_path,
                                          cudf::strings::get_json_object_options const& opts,
                                          rmm::mr::device_memory_resource* mr) {
  try {
    auto obj        = cudf::strings::get_json_object(view(), json_path, opts, mr);
    auto null_count = cudf::detail::count_unset_bits(
      obj->view().null_mask(), 0, obj->size(), rmm::cuda_stream_default);
    obj->set_null_count(null_count);
    return Column::New(Env(), std::move(obj));
  } catch (std::exception const& e) { NAPI_THROW(Napi::Error::New(Env(), e.what())); }
}

Napi::Value Column::get_json_object(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};
  return get_json_object(args[0], args[1], args[2]);
}

}  // namespace nv
