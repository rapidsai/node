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
#include "cudf/types.hpp"

#include <cudf/stream_compaction.hpp>
#include <cudf/table/table_view.hpp>

#include <memory>
#include <vector>

namespace nv {

std::vector<cudf::size_type> get_keys_vector(cudf::size_type numColumns) {
  std::vector<cudf::size_type> keys{0, numColumns - 1};
  return keys;
}

ObjectUnwrap<Table> Table::apply_boolean_mask(Column const& boolean_mask,
                                              rmm::mr::device_memory_resource* mr) const {
  return Table::New(cudf::apply_boolean_mask(cudf::table_view{{*this}}, boolean_mask, mr));
}

// ObjectUnwrap<Table> Table::drop_nulls(cudf::size_type threshold, rmm::mr::device_memory_resource*
// mr) const {
//   return Table::New(cudf::drop_nulls(cudf::table_view{{*this}},
//   get_keys_vector(this->num_columns()), threshold, mr));
// }

ObjectUnwrap<Table> Table::drop_nulls(rmm::mr::device_memory_resource* mr) const {
  return Table::New(
    cudf::drop_nulls(cudf::table_view{{*this}}, get_keys_vector(this->num_columns()), mr));
}

Napi::Value Table::drop_nulls(Napi::CallbackInfo const& info) {
  // if(info.Length() == 2){
  //   //threshold parameter not passed
  //   return drop_nulls(NapiToCPP(info[0]), NapiToCPP(info[1]).operator
  //   rmm::mr::device_memory_resource*())->Value();
  // }
  return drop_nulls(NapiToCPP(info[0]).operator rmm::mr::device_memory_resource*())->Value();
}

// ObjectUnwrap<Table> Table::drop_nans(cudf::size_type threshold, rmm::mr::device_memory_resource*
// mr) const {
//   return Table::New(cudf::drop_nans(cudf::table_view{{*this}},
//   get_keys_vector(this->num_columns()), threshold, mr));
// }

ObjectUnwrap<Table> Table::drop_nans(rmm::mr::device_memory_resource* mr) const {
  return Table::New(
    cudf::drop_nans(cudf::table_view{{*this}}, get_keys_vector(this->num_columns()), mr));
}

Napi::Value Table::drop_nans(Napi::CallbackInfo const& info) {
  // if(info.Length() == 2){
  //   //threshold parameter not passed
  //   return drop_nans(NapiToCPP(info[0]), NapiToCPP(info[1]).operator
  //   rmm::mr::device_memory_resource*())->Value();
  // }
  return drop_nans(NapiToCPP(info[0]).operator rmm::mr::device_memory_resource*())->Value();
}

}  // namespace nv
