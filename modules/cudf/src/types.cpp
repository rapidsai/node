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

#include "node_cudf/macros.hpp"

#include <cudf/types.hpp>

#include <napi.h>

namespace nv {

namespace types {
Napi::Object initModule(Napi::Env env, Napi::Object exports) {

    auto typeID = Napi::Object::New(env);
    EXPORT_ENUM(env, typeID, "EMPTY", cudf::type_id::EMPTY);
    EXPORT_ENUM(env, typeID, "INT8", cudf::type_id::INT8);
    EXPORT_ENUM(env, typeID, "INT16", cudf::type_id::INT16);
    EXPORT_ENUM(env, typeID, "INT32", cudf::type_id::INT32);
    EXPORT_ENUM(env, typeID, "INT64", cudf::type_id::INT64);
    EXPORT_ENUM(env, typeID, "UINT8", cudf::type_id::UINT8);
    EXPORT_ENUM(env, typeID, "UINT16", cudf::type_id::UINT16);
    EXPORT_ENUM(env, typeID, "UINT32", cudf::type_id::UINT32);
    EXPORT_ENUM(env, typeID, "UINT64", cudf::type_id::UINT64);
    EXPORT_ENUM(env, typeID, "FLOAT32", cudf::type_id::FLOAT32);
    EXPORT_ENUM(env, typeID, "FLOAT64", cudf::type_id::FLOAT64);
    EXPORT_ENUM(env, typeID, "BOOL8", cudf::type_id::BOOL8);
    EXPORT_ENUM(env, typeID, "TIMESTAMP_DAYS", cudf::type_id::TIMESTAMP_DAYS);
    EXPORT_ENUM(env, typeID, "TIMESTAMP_SECONDS", cudf::type_id::TIMESTAMP_SECONDS);
    EXPORT_ENUM(env, typeID, "TIMESTAMP_MILLISECONDS", cudf::type_id::TIMESTAMP_MILLISECONDS);
    EXPORT_ENUM(env, typeID, "TIMESTAMP_MICROSECONDS", cudf::type_id::TIMESTAMP_MICROSECONDS);
    EXPORT_ENUM(env, typeID, "TIMESTAMP_NANOSECONDS", cudf::type_id::TIMESTAMP_NANOSECONDS);
    EXPORT_ENUM(env, typeID, "DURATION_DAYS", cudf::type_id::DURATION_DAYS);
    EXPORT_ENUM(env, typeID, "DURATION_SECONDS", cudf::type_id::DURATION_SECONDS);
    EXPORT_ENUM(env, typeID, "DURATION_MILLISECONDS", cudf::type_id::DURATION_MILLISECONDS);
    EXPORT_ENUM(env, typeID, "DURATION_NANOSECONDS", cudf::type_id::DURATION_NANOSECONDS);
    EXPORT_ENUM(env, typeID, "DICTIONARY32", cudf::type_id::DICTIONARY32);
    EXPORT_ENUM(env, typeID, "STRING", cudf::type_id::STRING);
    EXPORT_ENUM(env, typeID, "LIST", cudf::type_id::LIST);
    EXPORT_ENUM(env, typeID, "DECIMAL32", cudf::type_id::DECIMAL32);
    EXPORT_ENUM(env, typeID, "DECIMAL64", cudf::type_id::DECIMAL64);

    EXPORT_PROP(exports, "typeID", typeID);
    return exports;
    }
}
}
