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

ObjectUnwrap<Table> Table::apply_boolean_mask(Column const& boolean_mask,
                                              rmm::mr::device_memory_resource* mr) const {
  return Table::New(cudf::apply_boolean_mask(cudf::table_view{{*this}}, boolean_mask, mr));
}

ObjectUnwrap<Table> Table::drop_nulls(std::vector<cudf::size_type> keys,
                                      cudf::size_type threshold,
                                      rmm::mr::device_memory_resource* mr) const {
  return Table::New(cudf::drop_nulls(*this, keys, threshold, mr));
}

Napi::Value Table::drop_nulls(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};
  return drop_nulls(args[0], args[1], args[2])->Value();
}

ObjectUnwrap<Table> Table::drop_nans(std::vector<cudf::size_type> keys,
                                     cudf::size_type threshold,
                                     rmm::mr::device_memory_resource* mr) const {
  return Table::New(cudf::drop_nans(*this, keys, threshold, mr));
}

Napi::Value Table::drop_nans(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};
  return drop_nans(args[0], args[1], args[2])->Value();
}

}  // namespace nv
