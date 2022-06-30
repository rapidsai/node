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

#include <node_rmm/device_buffer.hpp>

#include <cudf/column/column.hpp>
#include <cudf/filling.hpp>
#include <cudf/types.hpp>

#include <rmm/device_buffer.hpp>

#include <napi.h>

namespace nv {

Column::wrapper_t Column::fill(cudf::size_type begin,
                               cudf::size_type end,
                               cudf::scalar const& value,
                               rmm::mr::device_memory_resource* mr) {
  return Column::New(Env(), cudf::fill(*this, begin, end, value, mr));
}

Napi::Value Column::fill(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};
  auto scalar           = Scalar::Unwrap(args[0].ToObject());
  cudf::size_type begin = args.Length() > 1 ? args[1] : 0;
  cudf::size_type end   = args.Length() > 2 ? args[2] : size();
  try {
    return fill(begin, end, *scalar, args[3]);
  } catch (std::exception const& e) { throw Napi::Error::New(info.Env(), e.what()); }
}

void Column::fill_in_place(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};
  Scalar::wrapper_t scalar = args[0].ToObject();
  cudf::size_type begin    = args.Length() > 1 ? args[1] : 0;
  cudf::size_type end      = args.Length() > 2 ? args[2] : size();
  try {
    cudf::mutable_column_view view = *this;
    cudf::fill_in_place(view, begin, end, scalar->operator cudf::scalar&());
  } catch (std::exception const& e) { throw Napi::Error::New(info.Env(), e.what()); }
}

Column::wrapper_t Column::sequence(Napi::Env const& env,
                                   cudf::size_type size,
                                   cudf::scalar const& init,
                                   rmm::mr::device_memory_resource* mr) {
  try {
    return Column::New(env, cudf::sequence(size, init, mr));
  } catch (std::exception const& e) { throw Napi::Error::New(env, e.what()); }
}

Column::wrapper_t Column::sequence(Napi::Env const& env,
                                   cudf::size_type size,
                                   cudf::scalar const& init,
                                   cudf::scalar const& step,
                                   rmm::mr::device_memory_resource* mr) {
  try {
    return Column::New(env, cudf::sequence(size, init, step, mr));
  } catch (std::exception const& e) { throw Napi::Error::New(env, e.what()); }
}

Napi::Value Column::sequence(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};

  if (args.Length() != 3 and args.Length() != 4) {
    NAPI_THROW(Napi::Error::New(
      info.Env(), "sequence expects a size, init, and optionally a step and/or memory resource"));
  }

  if (!args[0].IsNumber()) {
    throw Napi::Error::New(info.Env(), "sequence size argument expects a number");
  }
  cudf::size_type size = args[0];

  if (!Scalar::IsInstance(args[1])) {
    throw Napi::Error::New(info.Env(), "sequence init argument expects a scalar");
  }

  Scalar::wrapper_t init = args[1].As<Napi::Object>();

  if (args.Length() == 3) {
    rmm::mr::device_memory_resource* mr = args[2];
    return Column::sequence(info.Env(), size, init, mr);
  } else {
    if (!Scalar::IsInstance(args[2])) {
      throw Napi::Error::New(info.Env(), "sequence step argument expects a scalar");
    }
    Scalar::wrapper_t step              = args[2].As<Napi::Object>();
    rmm::mr::device_memory_resource* mr = args[3];
    return Column::sequence(info.Env(), size, init, step, mr);
  }
}

}  // namespace nv
