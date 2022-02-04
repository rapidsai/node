// Copyright (c) 2021-2022, NVIDIA CORPORATION.
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
#include <node_cudf/utilities/dtypes.hpp>
#include <node_cudf/utilities/error.hpp>
#include <node_cudf/utilities/napi_to_cpp.hpp>

#include <cudf/column/column.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

namespace nv {

namespace {

template <typename LHS, typename RHS, typename = void>
struct common_type_exists : std::false_type {};

template <typename LHS, typename RHS>
struct common_type_exists<LHS, RHS, std::void_t<std::common_type_t<LHS, RHS>>> : std::true_type {};

struct get_common_type_id {
  template <typename LHS,
            typename RHS,
            typename std::enable_if_t<common_type_exists<LHS, RHS>::value>* = nullptr>
  cudf::type_id operator()() {
    return cudf::type_to_id<std::common_type_t<LHS, RHS>>();
  }
  template <typename LHS,
            typename RHS,
            typename std::enable_if_t<not common_type_exists<LHS, RHS>::value>* = nullptr>
  cudf::type_id operator()() {
    CUDF_FAIL("Cannot determine a logical common type between " +
              cudf::type_to_name{}.operator()<LHS>() + " and " +
              cudf::type_to_name{}.operator()<RHS>());
  }
};

}  // namespace

cudf::type_id get_common_type(cudf::data_type const& lhs, cudf::data_type const& rhs) {
  return cudf::double_type_dispatcher(lhs, rhs, get_common_type_id{});
}

Napi::Value find_common_type(CallbackArgs const& args) {
  try {
    return cudf_to_arrow_type(args.Env(), cudf::data_type{get_common_type(args[0], args[1])});
  } catch (cudf::logic_error const& err) { NODE_CUDF_THROW(err.what(), args.Env()); }
}

cudf::data_type arrow_to_cudf_type(Napi::Object const& type) {
  using cudf::data_type;
  using cudf::type_id;
  switch (type.Get("typeId").ToNumber().Int32Value()) {
    case -1 /*Arrow.Dictionary     */: return data_type{type_id::DICTIONARY32};
    case 0 /*Arrow.NONE            */: return data_type{type_id::EMPTY};
    case 1 /*Arrow.Null            */: return data_type{type_id::EMPTY};
    case 2 /*Arrow.Int             */: {
      switch (type.Get("bitWidth").ToNumber().Int32Value()) {
        case 8:
          return data_type{type.Get("isSigned").ToBoolean().Value() ? type_id::INT8
                                                                    : type_id::UINT8};
        case 16:
          return data_type{type.Get("isSigned").ToBoolean().Value() ? type_id::INT16
                                                                    : type_id::UINT16};
        case 32:
          return data_type{type.Get("isSigned").ToBoolean().Value() ? type_id::INT32
                                                                    : type_id::UINT32};
        case 64:
          return data_type{type.Get("isSigned").ToBoolean().Value() ? type_id::INT64
                                                                    : type_id::UINT64};
      }
      break;
    }
    case 3 /*Arrow.Float           */: {
      switch (type.Get("precision").ToNumber().Int32Value()) {
        // case 0 /*Arrow.HALF */: return data_type{type_id::FLOAT16};
        case 1 /*Arrow.SINGLE */: return data_type{type_id::FLOAT32};
        case 2 /*Arrow.DOUBLE */: return data_type{type_id::FLOAT64};
      }
      break;
    }
    case 4 /*Arrow.Binary          */: return data_type{type_id::STRING};
    case 5 /*Arrow.Utf8            */: return data_type{type_id::STRING};
    case 6 /*Arrow.Bool            */: return data_type{type_id::BOOL8};
    // case 7 /*Arrow.Decimal         */:
    case 8 /*Arrow.Date            */: {
      switch (type.Get("unit").ToNumber().Int32Value()) {
        case 0: return data_type{type_id::TIMESTAMP_DAYS};
        case 1: return data_type{type_id::TIMESTAMP_MILLISECONDS};
      }
    }
    // case 9 /*Arrow.Time            */:
    case 10 /*Arrow.Timestamp       */: {
      switch (type.Get("unit").ToNumber().Int32Value()) {
        case 0: return data_type{type_id::TIMESTAMP_SECONDS};
        case 1: return data_type{type_id::TIMESTAMP_MILLISECONDS};
        case 2: return data_type{type_id::TIMESTAMP_MICROSECONDS};
        case 3: return data_type{type_id::TIMESTAMP_NANOSECONDS};
      }
    }
    // case 11 /*Arrow.Interval        */:
    case 12 /*Arrow.List            */: return data_type{type_id::LIST};
    case 13 /*Arrow.Struct          */:
      return data_type{type_id::STRUCT};
      // case 14 /*Arrow.Union           */:
      // case 15 /*Arrow.FixedSizeBinary */:
      // case 16 /*Arrow.FixedSizeList   */:
      // case 17 /*Arrow.Map             */:
  }
  throw Napi::Error::New(
    type.Env(), "Unrecognized Arrow type '" + type.Get("typeId").ToString().Utf8Value() + "");
}

Napi::Object cudf_scalar_type_to_arrow_type(Napi::Env const& env, cudf::data_type type) {
  auto arrow_type = Napi::Object::New(env);
  switch (type.id()) {
    case cudf::type_id::EMPTY: {
      arrow_type.Set("typeId", 0);
      break;
    }
    case cudf::type_id::INT8: {
      arrow_type.Set("typeId", 2);
      arrow_type.Set("bitWidth", 8);
      arrow_type.Set("isSigned", true);
      break;
    }
    case cudf::type_id::INT16: {
      arrow_type.Set("typeId", 2);
      arrow_type.Set("bitWidth", 16);
      arrow_type.Set("isSigned", true);
      break;
    }
    case cudf::type_id::INT32: {
      arrow_type.Set("typeId", 2);
      arrow_type.Set("bitWidth", 32);
      arrow_type.Set("isSigned", true);
      break;
    }
    case cudf::type_id::INT64: {
      arrow_type.Set("typeId", 2);
      arrow_type.Set("bitWidth", 64);
      arrow_type.Set("isSigned", true);
      break;
    }
    case cudf::type_id::UINT8: {
      arrow_type.Set("typeId", 2);
      arrow_type.Set("bitWidth", 8);
      arrow_type.Set("isSigned", false);
      break;
    }
    case cudf::type_id::UINT16: {
      arrow_type.Set("typeId", 2);
      arrow_type.Set("bitWidth", 16);
      arrow_type.Set("isSigned", false);
      break;
    }
    case cudf::type_id::UINT32: {
      arrow_type.Set("typeId", 2);
      arrow_type.Set("bitWidth", 32);
      arrow_type.Set("isSigned", false);
      break;
    }
    case cudf::type_id::UINT64: {
      arrow_type.Set("typeId", 2);
      arrow_type.Set("bitWidth", 64);
      arrow_type.Set("isSigned", false);
      break;
    }
    case cudf::type_id::FLOAT32: {
      arrow_type.Set("typeId", 3);
      arrow_type.Set("precision", 1);
      break;
    }
    case cudf::type_id::FLOAT64: {
      arrow_type.Set("typeId", 3);
      arrow_type.Set("precision", 2);
      break;
    }
    case cudf::type_id::BOOL8: {
      arrow_type.Set("typeId", 6);
      break;
    }
    // case cudf::type_id::TIMESTAMP_DAYS: // TODO
    case cudf::type_id::TIMESTAMP_SECONDS: {
      arrow_type.Set("typeId", 10);
      arrow_type.Set("unit", 0);
      break;
    }
    case cudf::type_id::TIMESTAMP_MILLISECONDS: {
      arrow_type.Set("typeId", 10);
      arrow_type.Set("unit", 1);
      break;
    }
    case cudf::type_id::TIMESTAMP_MICROSECONDS: {
      arrow_type.Set("typeId", 10);
      arrow_type.Set("unit", 2);
      break;
    }
    case cudf::type_id::TIMESTAMP_NANOSECONDS: {
      arrow_type.Set("typeId", 10);
      arrow_type.Set("unit", 3);
      break;
    }

    // case cudf::type_id::DURATION_DAYS: // TODO
    // case cudf::type_id::DURATION_SECONDS: // TODO
    // case cudf::type_id::DURATION_MILLISECONDS: // TODO
    // case cudf::type_id::DURATION_MICROSECONDS: // TODO
    // case cudf::type_id::DURATION_NANOSECONDS: // TODO
    case cudf::type_id::STRING: {
      arrow_type.Set("typeId", 5);
      break;
    }
    // case cudf::type_id::DECIMAL32: // TODO
    // case cudf::type_id::DECIMAL64: // TODO
    default:
      throw Napi::Error::New(env,
                             "cudf_scalar_type_id_to_arrow_type not implemented for type: " +
                               cudf::type_dispatcher(type, cudf::type_to_name{}));
  }
  return arrow_type;
}

Napi::Object cudf_to_arrow_type(Napi::Env const& env, cudf::data_type const& cudf_type) {
  return cudf_scalar_type_to_arrow_type(env, cudf_type);
}

Napi::Object column_to_arrow_type(Napi::Env const& env,
                                  cudf::data_type const& cudf_type,
                                  Napi::Array children) {
  auto arrow_type = Napi::Object::New(env);
  switch (cudf_type.id()) {
    case cudf::type_id::DICTIONARY32: {
      arrow_type.Set("typeId", -1);
      arrow_type.Set("indices", children.Get(0u).ToObject().Get("type"));
      arrow_type.Set("dictionary", children.Get(1).ToObject().Get("type"));
      arrow_type.Set("isOrdered", false);
      break;
    }
    case cudf::type_id::LIST: {
      auto list_children = Napi::Array::New(env, 1);
      if (children.Length() > 1) {
        auto field = Napi::Object::New(env);
        field.Set("type", children.Get(1).ToObject().Get("type"));
        list_children.Set(0u, field);
      }
      arrow_type.Set("typeId", 12);
      arrow_type.Set("children", list_children);
      break;
    }
    case cudf::type_id::STRUCT: {
      auto struct_children = Napi::Array::New(env, children.Length());
      for (uint32_t i = 0; i < children.Length(); ++i) {
        Napi::HandleScope scope{env};
        auto field = Napi::Object::New(env);
        field.Set("type", children.Get(i).ToObject().Get("type"));
        struct_children.Set(i, field);
      }
      arrow_type.Set("typeId", 13);
      arrow_type.Set("children", struct_children);
      break;
    }
    default: return cudf_scalar_type_to_arrow_type(env, cudf_type);
  }
  return arrow_type;
}

}  // namespace nv
