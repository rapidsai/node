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
#include <node_cudf/utilities/napi_to_cpp.hpp>

#include <nv_node/utilities/args.hpp>
#include <nv_node/utilities/napi_to_cpp.hpp>

#include <cudf/stream_compaction.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <memory>
#include <vector>

namespace nv {

Table::wrapper_t Table::apply_boolean_mask(Column const& boolean_mask,
                                           rmm::mr::device_memory_resource* mr) const {
  try {
    return Table::New(Env(), cudf::apply_boolean_mask(cudf::table_view{{*this}}, boolean_mask, mr));
  } catch (std::exception const& e) { throw Napi::Error::New(Env(), e.what()); }
}

Table::wrapper_t Table::drop_nulls(std::vector<cudf::size_type> keys,
                                   cudf::size_type threshold,
                                   rmm::mr::device_memory_resource* mr) const {
  try {
    return Table::New(Env(), cudf::drop_nulls(*this, keys, threshold, mr));
  } catch (std::exception const& e) { throw Napi::Error::New(Env(), e.what()); }
}

Napi::Value Table::drop_nulls(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};
  return drop_nulls(args[0], args[1], args[2]);
}

Table::wrapper_t Table::drop_nans(std::vector<cudf::size_type> keys,
                                  cudf::size_type threshold,
                                  rmm::mr::device_memory_resource* mr) const {
  try {
    return Table::New(Env(), cudf::drop_nans(*this, keys, threshold, mr));
  } catch (std::exception const& e) { throw Napi::Error::New(Env(), e.what()); }
}

Napi::Value Table::drop_nans(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};
  return drop_nans(args[0], args[1], args[2]);
}

Table::wrapper_t Table::unique(std::vector<cudf::size_type> keys,
                               cudf::duplicate_keep_option keep,
                               bool is_nulls_equal,
                               rmm::mr::device_memory_resource* mr) const {
  cudf::null_equality nulls_equal =
    is_nulls_equal ? cudf::null_equality::EQUAL : cudf::null_equality::UNEQUAL;

  try {
    return Table::New(Env(), cudf::unique(*this, keys, keep, nulls_equal, mr));
  } catch (std::exception const& e) { throw Napi::Error::New(Env(), e.what()); }
}

Napi::Value Table::unique(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};
  return unique(args[0], args[1], args[2], args[3]);
}

Table::wrapper_t Table::distinct(std::vector<cudf::size_type> keys,
                                 bool is_nulls_equal,
                                 rmm::mr::device_memory_resource* mr) const {
  cudf::null_equality nulls_equal =
    is_nulls_equal ? cudf::null_equality::EQUAL : cudf::null_equality::UNEQUAL;
  cudf::nan_equality nans_equal =
    is_nulls_equal ? cudf::nan_equality::ALL_EQUAL : cudf::nan_equality::UNEQUAL;

  try {
    return Table::New(
      Env(),
      cudf::distinct(
        *this, keys, cudf::duplicate_keep_option::KEEP_ANY, nulls_equal, nans_equal, mr));
  } catch (std::exception const& e) { throw Napi::Error::New(Env(), e.what()); }
}

Napi::Value Table::distinct(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};
  return unique(args[0], args[1], args[2]);
}

}  // namespace nv
