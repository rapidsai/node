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

#include "node_cudf/types.hpp"
#include "node_cudf/utilities/cpp_to_napi.hpp"
#include "node_cudf/utilities/napi_to_cpp.hpp"

#include <cudf/types.hpp>
#include <node_cuda/utilities/error.hpp>
#include <nv_node/macros.hpp>
#include <nv_node/utilities/args.hpp>

#include <napi.h>

namespace nv {

Napi::FunctionReference DataType::constructor;

Napi::Object DataType::Init(Napi::Env env, Napi::Object exports) {
  Napi::Function ctor =
    DefineClass(env,
                "DataType",
                {
                  InstanceAccessor("id", &DataType::id, nullptr, napi_enumerable),
                });

  DataType::constructor = Napi::Persistent(ctor);
  DataType::constructor.SuppressDestruct();

  exports.Set("DataType", ctor);

  auto TypeId = Napi::Object::New(env);
  EXPORT_ENUM(env, TypeId, "EMPTY", cudf::type_id::EMPTY);
  EXPORT_ENUM(env, TypeId, "INT8", cudf::type_id::INT8);
  EXPORT_ENUM(env, TypeId, "INT16", cudf::type_id::INT16);
  EXPORT_ENUM(env, TypeId, "INT32", cudf::type_id::INT32);
  EXPORT_ENUM(env, TypeId, "INT64", cudf::type_id::INT64);
  EXPORT_ENUM(env, TypeId, "UINT8", cudf::type_id::UINT8);
  EXPORT_ENUM(env, TypeId, "UINT16", cudf::type_id::UINT16);
  EXPORT_ENUM(env, TypeId, "UINT32", cudf::type_id::UINT32);
  EXPORT_ENUM(env, TypeId, "UINT64", cudf::type_id::UINT64);
  EXPORT_ENUM(env, TypeId, "FLOAT32", cudf::type_id::FLOAT32);
  EXPORT_ENUM(env, TypeId, "FLOAT64", cudf::type_id::FLOAT64);
  EXPORT_ENUM(env, TypeId, "BOOL8", cudf::type_id::BOOL8);
  EXPORT_ENUM(env, TypeId, "TIMESTAMP_DAYS", cudf::type_id::TIMESTAMP_DAYS);
  EXPORT_ENUM(env, TypeId, "TIMESTAMP_SECONDS", cudf::type_id::TIMESTAMP_SECONDS);
  EXPORT_ENUM(env, TypeId, "TIMESTAMP_MILLISECONDS", cudf::type_id::TIMESTAMP_MILLISECONDS);
  EXPORT_ENUM(env, TypeId, "TIMESTAMP_MICROSECONDS", cudf::type_id::TIMESTAMP_MICROSECONDS);
  EXPORT_ENUM(env, TypeId, "TIMESTAMP_NANOSECONDS", cudf::type_id::TIMESTAMP_NANOSECONDS);
  EXPORT_ENUM(env, TypeId, "DURATION_DAYS", cudf::type_id::DURATION_DAYS);
  EXPORT_ENUM(env, TypeId, "DURATION_SECONDS", cudf::type_id::DURATION_SECONDS);
  EXPORT_ENUM(env, TypeId, "DURATION_MILLISECONDS", cudf::type_id::DURATION_MILLISECONDS);
  EXPORT_ENUM(env, TypeId, "DURATION_NANOSECONDS", cudf::type_id::DURATION_NANOSECONDS);
  EXPORT_ENUM(env, TypeId, "DICTIONARY32", cudf::type_id::DICTIONARY32);
  EXPORT_ENUM(env, TypeId, "STRING", cudf::type_id::STRING);
  EXPORT_ENUM(env, TypeId, "LIST", cudf::type_id::LIST);
  EXPORT_ENUM(env, TypeId, "DECIMAL32", cudf::type_id::DECIMAL32);
  EXPORT_ENUM(env, TypeId, "DECIMAL64", cudf::type_id::DECIMAL64);

  EXPORT_PROP(exports, "TypeId", TypeId);

  return exports;
}

Napi::Object DataType::New(cudf::type_id id) {
  auto inst = DataType::constructor.New({});
  DataType::Unwrap(inst)->Initialize(id);
  return inst;
}

DataType::DataType(CallbackArgs const& args) : Napi::ObjectWrap<DataType>(args) {
  NODE_CUDA_EXPECT(args.IsConstructCall(), "DataType constructor requires 'new'");
  if (args.Length() == 1) { Initialize(args[0]); }
}

void DataType::Initialize(cudf::type_id id) { id_ = id; }

Napi::Value DataType::id(Napi::CallbackInfo const& info) { return CPPToNapi(info)(id()); }

}  // namespace nv
