// Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>
#include <node_cudf/column.hpp>

#include <napi.h>
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/types.hpp>
#include <node_rmm/device_buffer.hpp>
#include <nv_node/utilities/wrap.hpp>
#include <rmm/device_buffer.hpp>

namespace nv {

namespace {

inline rmm::mr::device_memory_resource* get_mr(Napi::Value const& arg) {
  return MemoryResource::is_instance(arg) ? *MemoryResource::Unwrap(arg.ToObject())
                                          : rmm::mr::get_current_device_resource();
}

}  // namespace

ObjectUnwrap<Column> Column::nans_to_nulls(rmm::mr::device_memory_resource* mr) const {
  auto result                                         = cudf::nans_to_nulls(this->view(), mr);
  auto got                                            = cudf::table(cudf::table_view{{*this}});
  std::vector<std::unique_ptr<cudf::column>> contents = got.release();
  contents[0]->set_null_mask(std::move(*(result.first)));
  return Column::New(std::move(contents[0]));
}

Napi::Value Column::nans_to_nulls(Napi::CallbackInfo const& info) {
  return nans_to_nulls(get_mr(info[0]));
}

}  // namespace nv
