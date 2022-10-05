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
#include <cudf/strings/convert/convert_datetime.hpp>
#include <cudf/strings/convert/convert_floats.hpp>
#include <cudf/strings/convert/convert_integers.hpp>
#include <cudf/strings/convert/convert_ipv4.hpp>
#include <cudf/strings/convert/convert_lists.hpp>
#include <cudf/unary.hpp>

#include <rmm/mr/device/per_device_resource.hpp>

namespace nv {

namespace {
Column::wrapper_t lists_to_strings(
  Napi::Env const& env,
  cudf::lists_column_view const& input,
  std::string const& na_rep,
  cudf::strings_column_view const& separators,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()) {
  try {
    return Column::New(
      env, cudf::strings::format_list_column(input, cudf::string_scalar(na_rep), separators, mr));
  } catch (std::exception const& e) { throw Napi::Error::New(env, e.what()); }
}
}  // namespace

Column::wrapper_t Column::strings_from_booleans(rmm::mr::device_memory_resource* mr) const {
  try {
    return Column::New(
      Env(),
      cudf::strings::from_booleans(
        this->view(), cudf::string_scalar("true"), cudf::string_scalar("false"), mr));
  } catch (std::exception const& e) { throw Napi::Error::New(Env(), e.what()); }
}

Column::wrapper_t Column::strings_to_booleans(rmm::mr::device_memory_resource* mr) const {
  try {
    return Column::New(Env(),
                       cudf::strings::to_booleans(this->view(), cudf::string_scalar("true"), mr));
  } catch (std::exception const& e) { throw Napi::Error::New(Env(), e.what()); }
}

Column::wrapper_t Column::string_is_timestamp(std::string_view format,
                                              rmm::mr::device_memory_resource* mr) const {
  return Column::New(Env(), cudf::strings::is_timestamp(this->view(), format, mr));
}

Column::wrapper_t Column::strings_from_timestamps(std::string_view format,
                                                  rmm::mr::device_memory_resource* mr) const {
  try {
    return Column::New(
      Env(),
      cudf::strings::from_timestamps(this->view(),
                                     format,
                                     cudf::strings_column_view(cudf::column_view{
                                       cudf::data_type{cudf::type_id::STRING}, 0, nullptr}),
                                     mr));
  } catch (std::exception const& e) { throw Napi::Error::New(Env(), e.what()); }
}

Column::wrapper_t Column::strings_to_timestamps(cudf::data_type timestamp_type,
                                                std::string_view format,
                                                rmm::mr::device_memory_resource* mr) const {
  try {
    return Column::New(Env(),
                       cudf::strings::to_timestamps(this->view(), timestamp_type, format, mr));
  } catch (std::exception const& e) { throw Napi::Error::New(Env(), e.what()); }
}

Column::wrapper_t Column::string_is_float(rmm::mr::device_memory_resource* mr) const {
  return Column::New(Env(), cudf::strings::is_float(this->view(), mr));
}

Column::wrapper_t Column::strings_from_floats(rmm::mr::device_memory_resource* mr) const {
  try {
    return Column::New(Env(), cudf::strings::from_floats(this->view(), mr));
  } catch (std::exception const& e) { throw Napi::Error::New(Env(), e.what()); }
}

Column::wrapper_t Column::strings_to_floats(cudf::data_type out_type,
                                            rmm::mr::device_memory_resource* mr) const {
  try {
    return Column::New(Env(), cudf::strings::to_floats(this->view(), out_type, mr));
  } catch (std::exception const& e) { throw Napi::Error::New(Env(), e.what()); }
}

Column::wrapper_t Column::string_is_integer(rmm::mr::device_memory_resource* mr) const {
  return Column::New(Env(), cudf::strings::is_integer(this->view(), mr));
}

Column::wrapper_t Column::strings_from_integers(rmm::mr::device_memory_resource* mr) const {
  try {
    return Column::New(Env(), cudf::strings::from_integers(this->view(), mr));
  } catch (std::exception const& e) { throw Napi::Error::New(Env(), e.what()); }
}

Column::wrapper_t Column::strings_to_integers(cudf::data_type out_type,
                                              rmm::mr::device_memory_resource* mr) const {
  try {
    return Column::New(Env(), cudf::strings::to_integers(this->view(), out_type, mr));
  } catch (std::exception const& e) { throw Napi::Error::New(Env(), e.what()); }
}

Column::wrapper_t Column::string_is_hex(rmm::mr::device_memory_resource* mr) const {
  return Column::New(Env(), cudf::strings::is_hex(this->view(), mr));
}

Column::wrapper_t Column::hex_from_integers(rmm::mr::device_memory_resource* mr) const {
  try {
    return Column::New(Env(), cudf::strings::integers_to_hex(this->view(), mr));
  } catch (std::exception const& e) { throw Napi::Error::New(Env(), e.what()); }
}

Column::wrapper_t Column::hex_to_integers(cudf::data_type out_type,
                                          rmm::mr::device_memory_resource* mr) const {
  try {
    return Column::New(Env(), cudf::strings::hex_to_integers(this->view(), out_type, mr));
  } catch (std::exception const& e) { throw Napi::Error::New(Env(), e.what()); }
}

Column::wrapper_t Column::string_is_ipv4(rmm::mr::device_memory_resource* mr) const {
  return Column::New(Env(), cudf::strings::is_ipv4(this->view(), mr));
}

Column::wrapper_t Column::ipv4_from_integers(rmm::mr::device_memory_resource* mr) const {
  try {
    return Column::New(Env(), cudf::strings::integers_to_ipv4(this->view(), mr));
  } catch (std::exception const& e) { throw Napi::Error::New(Env(), e.what()); }
}

Column::wrapper_t Column::ipv4_to_integers(rmm::mr::device_memory_resource* mr) const {
  try {
    return Column::New(Env(), cudf::strings::ipv4_to_integers(this->view(), mr));
  } catch (std::exception const& e) { throw Napi::Error::New(Env(), e.what()); }
}

Napi::Value Column::strings_from_lists(Napi::CallbackInfo const& info) {
  CallbackArgs const args{info};
  std::string const na_rep            = args[0];
  Column::wrapper_t const separators  = args[1];
  rmm::mr::device_memory_resource* mr = args[2];
  return nv::lists_to_strings(info.Env(), view(), na_rep, separators->view(), mr);
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

Napi::Value Column::string_is_timestamp(Napi::CallbackInfo const& info) {
  CallbackArgs const args{info};
  std::string format = args[0];
  return string_is_timestamp(format, args[1]);
}

Napi::Value Column::strings_from_timestamps(Napi::CallbackInfo const& info) {
  CallbackArgs const args{info};
  std::string format = args[0];
  return strings_from_timestamps(format, args[1]);
}

Napi::Value Column::strings_to_timestamps(Napi::CallbackInfo const& info) {
  CallbackArgs const args{info};
  std::string format = args[1];
  return strings_to_timestamps(args[0], format, args[2]);
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
    NODE_CUDF_THROW("Column to_integers expects an output type and optional MemoryResource",
                    info.Env());
  }
  CallbackArgs const args{info};
  rmm::mr::device_memory_resource* mr = args[1];
  return strings_to_integers(args[0], mr);
}

Napi::Value Column::string_is_hex(Napi::CallbackInfo const& info) {
  CallbackArgs const args{info};
  rmm::mr::device_memory_resource* mr = args[0];
  return string_is_hex(mr);
}

Napi::Value Column::hex_from_integers(Napi::CallbackInfo const& info) {
  CallbackArgs const args{info};
  rmm::mr::device_memory_resource* mr = args[0];
  return hex_from_integers(mr);
}

Napi::Value Column::hex_to_integers(Napi::CallbackInfo const& info) {
  if (info.Length() < 1) {
    NODE_CUDF_THROW("Column hex_to_integers expects an output type and optional MemoryResource",
                    info.Env());
  }
  CallbackArgs const args{info};
  rmm::mr::device_memory_resource* mr = args[1];
  return hex_to_integers(args[0], mr);
}

Napi::Value Column::string_is_ipv4(Napi::CallbackInfo const& info) {
  CallbackArgs const args{info};
  rmm::mr::device_memory_resource* mr = args[0];
  return string_is_ipv4(mr);
}

Napi::Value Column::ipv4_from_integers(Napi::CallbackInfo const& info) {
  CallbackArgs const args{info};
  rmm::mr::device_memory_resource* mr = args[0];
  return ipv4_from_integers(mr);
}

Napi::Value Column::ipv4_to_integers(Napi::CallbackInfo const& info) {
  CallbackArgs const args{info};
  rmm::mr::device_memory_resource* mr = args[0];
  return ipv4_to_integers(mr);
}

}  // namespace nv
