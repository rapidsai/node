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

#include "node_cudf/scalar.hpp"
#include "node_cudf/types.hpp"
#include "node_cudf/utilities/cpp_to_napi.hpp"
#include "node_cudf/utilities/napi_to_cpp.hpp"

#include <node_cuda/utilities/error.hpp>
#include <node_cuda/utilities/napi_to_cpp.hpp>
#include <node_rmm/utilities/napi_to_cpp.hpp>

#include <napi.h>

namespace nv {

Napi::FunctionReference Scalar::constructor;

Napi::Object Scalar::Init(Napi::Env env, Napi::Object exports) {
  Napi::Function ctor = DefineClass(
    env,
    "Scalar",
    {
      InstanceAccessor("type", &Scalar::type, nullptr, napi_enumerable),
      InstanceAccessor("value", &Scalar::get_value, &Scalar::set_value, napi_enumerable),
    });

  Scalar::constructor = Napi::Persistent(ctor);
  Scalar::constructor.SuppressDestruct();
  exports.Set("Scalar", ctor);

  return exports;
}

Napi::Object Scalar::New(std::unique_ptr<cudf::scalar> scalar) {
  auto inst                     = Scalar::constructor.New({});
  Scalar::Unwrap(inst)->scalar_ = std::move(scalar);
  return inst;
}

Scalar::Scalar(CallbackArgs const& args) : Napi::ObjectWrap<Scalar>(args) {
  NODE_CUDA_EXPECT(args.IsConstructCall(), "Scalar constructor requires 'new'");
}

void Scalar::Finalize(Napi::Env env) { this->scalar_.reset(nullptr); }

Napi::Value Scalar::type(Napi::CallbackInfo const& info) {
  return DataType::New(this->type().id());
}

Napi::Value Scalar::get_value() {
  return this->is_valid(0) ? this->operator Napi::Value() : Env().Null();
}

Napi::Value Scalar::get_value(Napi::CallbackInfo const& info) { return get_value(); }

// TODO: Move these into type-dispatched functors

Scalar::operator Napi::Value() const {
  using namespace cudf;
  auto cast = CPPToNapi(Env());
  switch (type().id()) {
    case type_id::BOOL8: return cast(static_cast<numeric_scalar<bool>*>(scalar_.get())->value());
    case type_id::INT8: return cast(static_cast<numeric_scalar<int8_t>*>(scalar_.get())->value());
    case type_id::INT16: return cast(static_cast<numeric_scalar<int16_t>*>(scalar_.get())->value());
    case type_id::INT32: return cast(static_cast<numeric_scalar<int32_t>*>(scalar_.get())->value());
    case type_id::INT64: return cast(static_cast<numeric_scalar<int64_t>*>(scalar_.get())->value());
    case type_id::UINT8: return cast(static_cast<numeric_scalar<uint8_t>*>(scalar_.get())->value());
    case type_id::UINT16:
      return cast(static_cast<numeric_scalar<uint16_t>*>(scalar_.get())->value());
    case type_id::UINT32:
      return cast(static_cast<numeric_scalar<uint32_t>*>(scalar_.get())->value());
    case type_id::UINT64:
      return cast(static_cast<numeric_scalar<uint64_t>*>(scalar_.get())->value());
    case type_id::FLOAT32: return cast(static_cast<numeric_scalar<float>*>(scalar_.get())->value());
    case type_id::FLOAT64:
      return cast(static_cast<numeric_scalar<double>*>(scalar_.get())->value());
    case type_id::STRING: return cast(static_cast<string_scalar*>(scalar_.get())->to_string());
    case type_id::TIMESTAMP_DAYS:
      return cast(static_cast<timestamp_scalar<timestamp_D>*>(scalar_.get())->value());
    case type_id::TIMESTAMP_SECONDS:
      return cast(static_cast<timestamp_scalar<timestamp_s>*>(scalar_.get())->value());
    case type_id::TIMESTAMP_MILLISECONDS:
      return cast(static_cast<timestamp_scalar<timestamp_ms>*>(scalar_.get())->value());
    case type_id::TIMESTAMP_MICROSECONDS:
      return cast(static_cast<timestamp_scalar<timestamp_us>*>(scalar_.get())->value());
    case type_id::TIMESTAMP_NANOSECONDS:
      return cast(static_cast<timestamp_scalar<timestamp_ns>*>(scalar_.get())->value());
    case type_id::DURATION_DAYS:
      return cast(static_cast<duration_scalar<duration_D>*>(scalar_.get())->value());
    case type_id::DURATION_SECONDS:
      return cast(static_cast<duration_scalar<duration_s>*>(scalar_.get())->value());
    case type_id::DURATION_MILLISECONDS:
      return cast(static_cast<duration_scalar<duration_ms>*>(scalar_.get())->value());
    case type_id::DURATION_MICROSECONDS:
      return cast(static_cast<duration_scalar<duration_us>*>(scalar_.get())->value());
    case type_id::DURATION_NANOSECONDS:
      return cast(static_cast<duration_scalar<duration_ns>*>(scalar_.get())->value());
    // TODO
    case type_id::DICTIONARY32: return Env().Null();
    // TODO
    case type_id::LIST: return Env().Null();
    case type_id::DECIMAL32:
      return cast(static_cast<fixed_point_scalar<numeric::decimal32>*>(scalar_.get())->value(0));
    case type_id::DECIMAL64:
      return cast(static_cast<fixed_point_scalar<numeric::decimal64>*>(scalar_.get())->value(0));
    // TODO
    case type_id::STRUCT: return Env().Null();
    default: return Env().Null();
  }
}

void Scalar::set_value(Napi::CallbackInfo const& info, Napi::Value const& value) {
  if (value.IsNull() or value.IsUndefined()) {
    this->set_valid(false);
  } else {
    using namespace cudf;
    switch (type().id()) {
      case type_id::BOOL8:
        static_cast<numeric_scalar<bool>*>(scalar_.get())->set_value(NapiToCPP(value));
        break;
      case type_id::INT8:
        static_cast<numeric_scalar<int8_t>*>(scalar_.get())->set_value(NapiToCPP(value));
        break;
      case type_id::INT16:
        static_cast<numeric_scalar<int16_t>*>(scalar_.get())->set_value(NapiToCPP(value));
        break;
      case type_id::INT32:
        static_cast<numeric_scalar<int32_t>*>(scalar_.get())->set_value(NapiToCPP(value));
        break;
      case type_id::INT64:
        static_cast<numeric_scalar<int64_t>*>(scalar_.get())->set_value(NapiToCPP(value));
        break;
      case type_id::UINT8:
        static_cast<numeric_scalar<uint8_t>*>(scalar_.get())->set_value(NapiToCPP(value));
        break;
      case type_id::UINT16:
        static_cast<numeric_scalar<uint16_t>*>(scalar_.get())->set_value(NapiToCPP(value));
        break;
      case type_id::UINT32:
        static_cast<numeric_scalar<uint32_t>*>(scalar_.get())->set_value(NapiToCPP(value));
        break;
      case type_id::UINT64:
        static_cast<numeric_scalar<uint64_t>*>(scalar_.get())->set_value(NapiToCPP(value));
        break;
      case type_id::FLOAT32:
        static_cast<numeric_scalar<float>*>(scalar_.get())->set_value(NapiToCPP(value));
        break;
      case type_id::FLOAT64:
        static_cast<numeric_scalar<double>*>(scalar_.get())->set_value(NapiToCPP(value));
        break;
      case type_id::STRING: scalar_.reset(new string_scalar(value.ToString(), true)); break;
      case type_id::TIMESTAMP_DAYS:
        static_cast<timestamp_scalar<timestamp_D>*>(scalar_.get())->set_value(NapiToCPP(value));
        break;
      case type_id::TIMESTAMP_SECONDS:
        static_cast<timestamp_scalar<timestamp_s>*>(scalar_.get())->set_value(NapiToCPP(value));
        break;
      case type_id::TIMESTAMP_MILLISECONDS:
        static_cast<timestamp_scalar<timestamp_ms>*>(scalar_.get())->set_value(NapiToCPP(value));
        break;
      case type_id::TIMESTAMP_MICROSECONDS:
        static_cast<timestamp_scalar<timestamp_us>*>(scalar_.get())->set_value(NapiToCPP(value));
        break;
      case type_id::TIMESTAMP_NANOSECONDS:
        static_cast<timestamp_scalar<timestamp_ns>*>(scalar_.get())->set_value(NapiToCPP(value));
        break;
      case type_id::DURATION_DAYS:
        static_cast<duration_scalar<duration_D>*>(scalar_.get())->set_value(NapiToCPP(value));
        break;
      case type_id::DURATION_SECONDS:
        static_cast<duration_scalar<duration_s>*>(scalar_.get())->set_value(NapiToCPP(value));
        break;
      case type_id::DURATION_MILLISECONDS:
        static_cast<duration_scalar<duration_ms>*>(scalar_.get())->set_value(NapiToCPP(value));
        break;
      case type_id::DURATION_MICROSECONDS:
        static_cast<duration_scalar<duration_us>*>(scalar_.get())->set_value(NapiToCPP(value));
        break;
      case type_id::DURATION_NANOSECONDS:
        static_cast<duration_scalar<duration_ns>*>(scalar_.get())->set_value(NapiToCPP(value));
        break;
      // TODO
      case type_id::DICTIONARY32: break;
      // TODO
      case type_id::LIST: break;
      case type_id::DECIMAL32:
        scalar_.reset(new fixed_point_scalar<numeric::decimal32>(value.ToNumber(), true));
        break;
      case type_id::DECIMAL64:
        scalar_.reset(new fixed_point_scalar<numeric::decimal64>(value.ToNumber(), true));
        break;
      // TODO
      case type_id::STRUCT: break;
      default: break;
    }
  }
}

}  // namespace nv
