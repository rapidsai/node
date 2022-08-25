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

#pragma once

#include <nv_node/utilities/args.hpp>

#include <cudf/types.hpp>

#include <arrow/type.h>

namespace nv {

cudf::type_id get_common_type(cudf::data_type const& lhs, cudf::data_type const& rhs);

cudf::data_type arrow_to_cudf_type(Napi::Object const& arrow_type);

Napi::Object column_to_arrow_type(Napi::Env const& env,
                                  cudf::data_type const& cudf_type,
                                  Napi::Array children);

Napi::Object cudf_to_arrow_type(Napi::Env const& env, cudf::data_type const& cudf_type);

Napi::Object cudf_scalar_type_to_arrow_type(Napi::Env const& env, cudf::data_type type);

Napi::Value find_common_type(CallbackArgs const& args);

}  // namespace nv

namespace Napi {

template <>
inline Value Value::From(napi_env env, cudf::data_type const& type) {
  return nv::cudf_to_arrow_type(env, type);
}

template <>
inline Value Value::From(napi_env env, arrow::DataType const& type);

template <>
inline Value Value::From(napi_env env, std::vector<std::shared_ptr<arrow::Field>> const& fields) {
  auto n  = fields.size();
  auto fs = Array::New(env, n);
  for (std::size_t i = 0; i < n; ++i) {
    auto f        = fields[i];
    auto o        = Object::New(env);
    o["name"]     = String::New(env, f->name());
    o["type"]     = Value::From(env, *f->type());
    o["nullable"] = Boolean::New(env, f->nullable());
    fs[i]         = o;
  }
  return fs;
}

template <>
inline Value Value::From(napi_env env, arrow::DataType const& type) {
  auto o = Napi::Object::New(env);
  switch (type.id()) {
    case arrow::Type::DICTIONARY: {  //
      o["typeId"]    = -1;
      o["idOrdered"] = dynamic_cast<arrow::DictionaryType const&>(type).ordered();
      o["indices"] =
        Value::From(env, *dynamic_cast<arrow::DictionaryType const&>(type).index_type());
      o["dictionary"] =
        Value::From(env, *dynamic_cast<arrow::DictionaryType const&>(type).value_type());
      return o;
    }
    case arrow::Type::INT8: {
      o["typeId"]   = 2;
      o["bitWidth"] = 8;
      o["isSigned"] = true;
      return o;
    }
    case arrow::Type::INT16: {
      o["typeId"]   = 2;
      o["bitWidth"] = 16;
      o["isSigned"] = true;
      return o;
    }
    case arrow::Type::INT32: {
      o["typeId"]   = 2;
      o["bitWidth"] = 32;
      o["isSigned"] = true;
      return o;
    }
    case arrow::Type::INT64: {
      o["typeId"]   = 2;
      o["bitWidth"] = 64;
      o["isSigned"] = true;
      return o;
    }
    case arrow::Type::UINT8: {
      o["typeId"]   = 2;
      o["bitWidth"] = 8;
      o["isSigned"] = false;
      return o;
    }
    case arrow::Type::UINT16: {
      o["typeId"]   = 2;
      o["bitWidth"] = 16;
      o["isSigned"] = false;
      return o;
    }
    case arrow::Type::UINT32: {
      o["typeId"]   = 2;
      o["bitWidth"] = 32;
      o["isSigned"] = false;
      return o;
    }
    case arrow::Type::UINT64: {
      o["typeId"]   = 2;
      o["bitWidth"] = 64;
      o["isSigned"] = false;
      return o;
    }
    case arrow::Type::FLOAT: {
      o["typeId"]    = 3;
      o["precision"] = 1;
      return o;
    }
    case arrow::Type::DOUBLE: {
      o["typeId"]    = 3;
      o["precision"] = 2;
      return o;
    }
    case arrow::Type::STRING: {
      o["typeId"] = 5;
      return o;
    }
    case arrow::Type::BOOL: {
      o["typeId"] = 6;
      return o;
    }
    case arrow::Type::TIMESTAMP: {
      o["typeId"] = 10;
      o["unit"]   = static_cast<int32_t>(dynamic_cast<arrow::TimestampType const&>(type).unit());
      o["bitWidth"] =
        static_cast<int32_t>(dynamic_cast<arrow::TimestampType const&>(type).bit_width());
      return o;
    }
    case arrow::Type::LIST: {
      o["typeId"]   = 12;
      o["children"] = Value::From(env, dynamic_cast<arrow::ListType const&>(type).fields());
      return o;
    }
    case arrow::Type::STRUCT: {
      o["typeId"]   = 13;
      o["children"] = Value::From(env, dynamic_cast<arrow::StructType const&>(type).fields());
      return o;
    }
    default: {
      throw Napi::Error::New(env, "Unrecognized Arrow type '" + type.ToString() + "");
    }
  }
  return o;
}

}  // namespace Napi
