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

#include <node_cudf/column.hpp>
#include <node_rmm/device_buffer.hpp>
#include <nv_node/utilities/wrap.hpp>

#include <napi.h>
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>
#include <rmm/device_buffer.hpp>

namespace nv {

std::pair<std::unique_ptr<rmm::device_buffer>, cudf::size_type> Column::nans_to_nulls(
  rmm::mr::device_memory_resource* mr) const {
  auto result = cudf::nans_to_nulls(*this, mr);
  return {std::move(result.first), result.second};
}

Napi::Value Column::nans_to_nulls(Napi::CallbackInfo const& info) {
  rmm::mr::device_memory_resource* mr = NapiToCPP(info[0]);
  auto result                         = nans_to_nulls(mr);

  std::unique_ptr<cudf::column> new_col =
    cudf::allocate_like(*this, cudf::mask_allocation_policy::NEVER, mr);
  auto new_col_view = new_col->mutable_view();
  cudf::copy_range_in_place(*this, new_col_view, 0, size(), 0);
  new_col->set_null_mask(std::move(*result.first));
  return Column::New(std::move(new_col));

}  // namespace nv
}  // namespace nv
