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
#include <node_rmm/device_buffer.hpp>
#include <nv_node/utilities/wrap.hpp>

#include <napi.h>
#include <cudf/column/column.hpp>
#include <cudf/filling.hpp>
#include <cudf/types.hpp>
#include <rmm/device_buffer.hpp>

namespace nv {

ObjectUnwrap<Column> Column::sequence(Napi::Env const& env,
                                      cudf::size_type size,
                                      cudf::scalar const& init,
                                      rmm::mr::device_memory_resource* mr) {
  try {
    return Column::New(cudf::sequence(size, init, mr));
  } catch (cudf::logic_error const& err) { NAPI_THROW(Napi::Error::New(env, err.what())); }
}

ObjectUnwrap<Column> Column::sequence(Napi::Env const& env,
                                      cudf::size_type size,
                                      cudf::scalar const& init,
                                      cudf::scalar const& step,
                                      rmm::mr::device_memory_resource* mr) {
  try {
    return Column::New(cudf::sequence(size, init, step, mr));
  } catch (cudf::logic_error const& err) { NAPI_THROW(Napi::Error::New(env, err.what())); }
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

  if (!Scalar::is_instance(args[1])) {
    throw Napi::Error::New(info.Env(), "sequence init argument expects a scalar");
  }
  auto& init = *Scalar::Unwrap(args[1]);

  if (args.Length() == 3) {
    rmm::mr::device_memory_resource* mr = args[2];
    return Column::sequence(info.Env(), size, init, mr);
  } else {
    if (!Scalar::is_instance(args[2])) {
      throw Napi::Error::New(info.Env(), "sequence step argument expects a scalar");
    }
    auto& step                          = *Scalar::Unwrap(args[2]);
    rmm::mr::device_memory_resource* mr = args[3];
    return Column::sequence(info.Env(), size, init, step, mr);
  }

}  // namespace nv
}  // namespace nv
