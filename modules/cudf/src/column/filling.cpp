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
#include <node_cudf/scalar.hpp>

#include <cudf/filling.hpp>

namespace nv {

ObjectUnwrap<Column> Column::fill(cudf::size_type begin,
                                  cudf::size_type end,
                                  cudf::scalar const& value,
                                  rmm::mr::device_memory_resource* mr) {
  return Column::New(cudf::fill(*this, begin, end, value, mr));
}

Napi::Value Column::fill(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};
  auto scalar           = Scalar::Unwrap(args[0].ToObject());
  cudf::size_type begin = args.Length() > 1 ? args[1] : 0;
  cudf::size_type end   = args.Length() > 2 ? args[2] : size();
  try {
    return fill(begin, end, *scalar, args[3]);
  } catch (cudf::logic_error const& e) { NAPI_THROW(Napi::Error::New(info.Env(), e.what())); }
}

void Column::fill_in_place(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};
  auto scalar           = Scalar::Unwrap(args[0].ToObject());
  cudf::size_type begin = args.Length() > 1 ? args[1] : 0;
  cudf::size_type end   = args.Length() > 2 ? args[2] : size();
  try {
    cudf::mutable_column_view view = *this;
    cudf::fill_in_place(view, begin, end, *scalar);
  } catch (cudf::logic_error const& e) { NAPI_THROW(Napi::Error::New(info.Env(), e.what())); }
}

}  // namespace nv
