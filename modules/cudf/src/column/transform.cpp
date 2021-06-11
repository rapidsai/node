// Copyright (c) 2020-2021, NVIDIA CORPORATION.
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
  try {
    return cudf::nans_to_nulls(*this, mr);
  } catch (std::exception const& e) { NAPI_THROW(Napi::Error::New(Env(), e.what())); }
}

Napi::Value Column::nans_to_nulls(Napi::CallbackInfo const& info) {
  rmm::mr::device_memory_resource* mr = NapiToCPP(info[0]);
  auto result                         = nans_to_nulls(mr);
  try {
    auto col =
      Column::New(Env(), cudf::allocate_like(*this, cudf::mask_allocation_policy::RETAIN, mr));

    [&](cudf::mutable_column_view view) {
      cudf::copy_range_in_place(*this, view, 0, size(), 0);
    }(col->mutable_view());

    col->set_null_mask(DeviceBuffer::New(info.Env(), std::move(result.first)),
                       null_count() == cudf::UNKNOWN_NULL_COUNT ? cudf::UNKNOWN_NULL_COUNT
                                                                : null_count() + result.second);

    return col;
  } catch (std::exception const& e) { NAPI_THROW(Napi::Error::New(Env(), e.what())); }
}  // namespace nv
}  // namespace nv
