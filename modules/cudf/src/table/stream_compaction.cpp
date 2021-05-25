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
#include <node_cudf/table.hpp>
#include <node_cudf/utilities/napi_to_cpp.hpp>
#include <nv_node/utilities/args.hpp>
#include <nv_node/utilities/napi_to_cpp.hpp>

#include "cudf/types.hpp"

#include <cudf/stream_compaction.hpp>
#include <cudf/table/table_view.hpp>

#include <memory>
#include <vector>

namespace nv {

Table::wrapper_t Table::apply_boolean_mask(Column const& boolean_mask,
                                           rmm::mr::device_memory_resource* mr) const {
  return Table::New(Env(), cudf::apply_boolean_mask(cudf::table_view{{*this}}, boolean_mask, mr));
}

Table::wrapper_t Table::drop_nulls(std::vector<cudf::size_type> keys,
                                   cudf::size_type threshold,
                                   rmm::mr::device_memory_resource* mr) const {
  return Table::New(Env(), cudf::drop_nulls(*this, keys, threshold, mr));
}

Napi::Value Table::drop_nulls(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};
  return drop_nulls(args[0], args[1], args[2]);
}

Table::wrapper_t Table::drop_nans(std::vector<cudf::size_type> keys,
                                  cudf::size_type threshold,
                                  rmm::mr::device_memory_resource* mr) const {
  return Table::New(Env(), cudf::drop_nans(*this, keys, threshold, mr));
}

Napi::Value Table::drop_nans(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};
  return drop_nans(args[0], args[1], args[2]);
}

Table::wrapper_t Table::drop_duplicates(cudf::duplicate_keep_option keep,
                                        bool is_nulls_equal,
                                        bool is_nulls_first,
                                        rmm::mr::device_memory_resource* mr) const {
  cudf::null_equality nulls_equal =
    is_nulls_equal ? cudf::null_equality::EQUAL : cudf::null_equality::UNEQUAL;
  cudf::null_order nulls_first =
    is_nulls_first ? cudf::null_order::BEFORE : cudf::null_order::AFTER;

  try {
    return Table::New(Env(), cudf::drop_duplicates(*this, {0}, keep, nulls_equal, nulls_first, mr));
  } catch (cudf::logic_error const& e) { NAPI_THROW(Napi::Error::New(Env(), e.what())); }
}

Napi::Value Table::drop_duplicates(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};
  return drop_duplicates(args[0], args[1], args[2], args[3]);
}

}  // namespace nv
