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

#include <nv_node/utilities/cpp_to_napi.hpp>

#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <napi.h>
#include <type_traits>

namespace nv {

namespace {

struct set_scalar_value {
  Napi::Value val;

  template <typename T>
  inline std::enable_if_t<cudf::is_numeric<T>(), void> operator()(
    std::unique_ptr<cudf::scalar>& scalar, cudaStream_t stream = 0) {
    static_cast<cudf::numeric_scalar<T>*>(scalar.get())->set_value(NapiToCPP(val), stream);
  }
  template <typename T>
  inline std::enable_if_t<std::is_same<T, cudf::string_view>::value, void> operator()(
    std::unique_ptr<cudf::scalar>& scalar, cudaStream_t stream = 0) {
    scalar.reset(new cudf::string_scalar(val.ToString(), true, stream));
  }
  template <typename T>
  inline std::enable_if_t<cudf::is_duration<T>(), void> operator()(
    std::unique_ptr<cudf::scalar>& scalar, cudaStream_t stream = 0) {
    static_cast<cudf::duration_scalar<T>*>(scalar.get())->set_value(NapiToCPP(val), stream);
  }
  template <typename T>
  inline std::enable_if_t<cudf::is_timestamp<T>(), void> operator()(
    std::unique_ptr<cudf::scalar>& scalar, cudaStream_t stream = 0) {
    static_cast<cudf::timestamp_scalar<T>*>(scalar.get())->set_value(NapiToCPP(val), stream);
  }
  template <typename T>
  inline std::enable_if_t<cudf::is_fixed_point<T>(), void> operator()(
    std::unique_ptr<cudf::scalar>& scalar, cudaStream_t stream = 0) {
    scalar.reset(new cudf::fixed_point_scalar<T>(val.ToNumber(), true, stream));
  }
  template <typename T>
  inline std::enable_if_t<!(cudf::is_numeric<T>() ||                      //
                            std::is_same<T, cudf::string_view>::value ||  //
                            cudf::is_duration<T>() ||                     //
                            cudf::is_timestamp<T>() ||                    //
                            cudf::is_fixed_point<T>()),
                          void>
  operator()(std::unique_ptr<cudf::scalar> const& scalar, cudaStream_t stream = 0) {
    NAPI_THROW(Napi::Error::New(val.Env(), "Unsupported dtype"));
  }
};

}  // namespace

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

ObjectUnwrap<Scalar> Scalar::New(std::unique_ptr<cudf::scalar> scalar) {
  auto opts = Napi::Object::New(constructor.Env());
  opts.Set("type", DataType::New(scalar->type().id())->Value());
  ObjectUnwrap<Scalar> inst{constructor.New({opts})};
  inst->scalar_ = std::move(scalar);
  return inst;
}

ObjectUnwrap<Scalar> Scalar::New(Napi::Number const& value) {
  return New(value, cudf::data_type{cudf::type_id::FLOAT64});
}

ObjectUnwrap<Scalar> Scalar::New(Napi::BigInt const& value) {
  return New(value, cudf::data_type{cudf::type_id::UINT64});
}

ObjectUnwrap<Scalar> Scalar::New(Napi::String const& value) {
  return New(value, cudf::data_type{cudf::type_id::STRING});
}

ObjectUnwrap<Scalar> Scalar::New(Napi::Value const& value, cudf::data_type type) {
  auto opts = Napi::Object::New(constructor.Env());
  opts.Set("value", value);
  opts.Set("type", DataType::New(type.id())->Value());
  return constructor.New({opts});
}

Scalar::Scalar(CallbackArgs const& args) : Napi::ObjectWrap<Scalar>(args) {
  NODE_CUDA_EXPECT(args[0].IsObject(), "Scalar constructor expects an Object options argument");
  Napi::Object props = args[0];

  NODE_CUDA_EXPECT(props.Has("type"), "Scalar constructor expects options to have a 'type' field");
  auto type = props.Get("type");

  NODE_CUDA_EXPECT(DataType::is_instance(type),
                   "Scalar constructor expects 'type' option to be a DataType");
  type_   = Napi::Persistent(type.ToObject());
  scalar_ = cudf::make_default_constructed_scalar(this->type());
  if (props.Has("value")) { set_value(args, props.Get("value")); }
}

void Scalar::Finalize(Napi::Env env) { this->scalar_.reset(nullptr); }

Napi::Value Scalar::type(Napi::CallbackInfo const& info) { return type_.Value(); }

Napi::Value Scalar::get_value() const { return CPPToNapi(Env())(scalar_); }

Scalar::operator Napi::Value() const { return get_value(); }

Scalar::operator cudf::scalar&() const { return *scalar_; }

Napi::Value Scalar::get_value(Napi::CallbackInfo const& info) { return get_value(); }

void Scalar::set_value(Napi::CallbackInfo const& info, Napi::Value const& value) {
  if (value.IsNull() or value.IsUndefined()) {
    this->set_valid(false);
  } else {
    cudf::type_dispatcher(this->type(), set_scalar_value{value}, scalar_);
  }
}

}  // namespace nv
