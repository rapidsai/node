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

#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/convert/convert_booleans.hpp>
#include <cudf/strings/convert/convert_floats.hpp>
#include <cudf/strings/convert/convert_integers.hpp>
#include <cudf/unary.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

namespace nv {

Column::wrapper_t Column::strings_from_booleans(rmm::mr::device_memory_resource* mr) const {
  try {
    return Column::New(
      Env(),
      cudf::strings::from_booleans(
        this->view(), cudf::string_scalar("true"), cudf::string_scalar("false"), mr));
  } catch (cudf::logic_error const& err) { NAPI_THROW(Napi::Error::New(Env(), err.what())); }
}

Column::wrapper_t Column::strings_to_booleans(rmm::mr::device_memory_resource* mr) const {
  try {
    return Column::New(Env(),
                       cudf::strings::to_booleans(this->view(), cudf::string_scalar("true"), mr));
  } catch (cudf::logic_error const& err) { NAPI_THROW(Napi::Error::New(Env(), err.what())); }
}

Column::wrapper_t Column::string_is_float(rmm::mr::device_memory_resource* mr) const {
  return Column::New(Env(), cudf::strings::is_float(this->view(), mr));
}

Column::wrapper_t Column::strings_from_floats(rmm::mr::device_memory_resource* mr) const {
  try {
    return Column::New(Env(), cudf::strings::from_floats(this->view(), mr));
  } catch (cudf::logic_error const& err) { NAPI_THROW(Napi::Error::New(Env(), err.what())); }
}

Column::wrapper_t Column::strings_to_floats(cudf::data_type out_type,
                                            rmm::mr::device_memory_resource* mr) const {
  try {
    return Column::New(Env(), cudf::strings::to_floats(this->view(), out_type, mr));
  } catch (cudf::logic_error const& err) { NAPI_THROW(Napi::Error::New(Env(), err.what())); }
}

Column::wrapper_t Column::string_is_integer(rmm::mr::device_memory_resource* mr) const {
  return Column::New(Env(), cudf::strings::is_integer(this->view(), mr));
}

Column::wrapper_t Column::strings_from_integers(rmm::mr::device_memory_resource* mr) const {
  try {
    return Column::New(Env(), cudf::strings::from_integers(this->view(), mr));
  } catch (cudf::logic_error const& err) { NAPI_THROW(Napi::Error::New(Env(), err.what())); }
}

Column::wrapper_t Column::strings_to_integers(cudf::data_type out_type,
                                              rmm::mr::device_memory_resource* mr) const {
  try {
    return Column::New(Env(), cudf::strings::to_integers(this->view(), out_type, mr));
  } catch (cudf::logic_error const& err) { NAPI_THROW(Napi::Error::New(Env(), err.what())); }
}

Napi::Value Column::strings_from_booleans(Napi::CallbackInfo const& info) {
  CallbackArgs const args{info};
  rmm::mr::device_memory_resource* mr = args[0];
  return strings_from_booleans(mr);
}

Napi::Value Column::strings_to_booleans(Napi::CallbackInfo const& info) {
  CallbackArgs const args{info};
  rmm::mr::device_memory_resource* mr = args[0];
  return strings_to_booleans(mr);
}

Napi::Value Column::string_is_float(Napi::CallbackInfo const& info) {
  CallbackArgs const args{info};
  rmm::mr::device_memory_resource* mr = args[0];
  return string_is_float(mr);
}

Napi::Value Column::strings_from_floats(Napi::CallbackInfo const& info) {
  CallbackArgs const args{info};
  rmm::mr::device_memory_resource* mr = args[0];
  return strings_from_floats(mr);
}

Napi::Value Column::strings_to_floats(Napi::CallbackInfo const& info) {
  if (info.Length() < 1) {
    NODE_CUDF_THROW("Column to_float expects an output type and optional MemoryResource",
                    info.Env());
  }
  CallbackArgs const args{info};
  rmm::mr::device_memory_resource* mr = args[1];
  return strings_to_floats(args[0], mr);
}

Napi::Value Column::string_is_integer(Napi::CallbackInfo const& info) {
  CallbackArgs const args{info};
  rmm::mr::device_memory_resource* mr = args[0];
  return string_is_integer(mr);
}

Napi::Value Column::strings_from_integers(Napi::CallbackInfo const& info) {
  CallbackArgs const args{info};
  rmm::mr::device_memory_resource* mr = args[0];
  return strings_from_integers(mr);
}

Napi::Value Column::strings_to_integers(Napi::CallbackInfo const& info) {
  if (info.Length() < 1) {
    NODE_CUDF_THROW("Column to_integer expects an output type and optional MemoryResource",
                    info.Env());
  }
  CallbackArgs const args{info};
  rmm::mr::device_memory_resource* mr = args[1];
  return strings_to_integers(args[0], mr);
}

}  // namespace nv
