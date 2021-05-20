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

#include <node_cudf/scalar.hpp>
#include <node_cudf/utilities/cpp_to_napi.hpp>
#include <node_cudf/utilities/dtypes.hpp>
#include <node_cudf/utilities/napi_to_cpp.hpp>
#include <node_cudf/utilities/value_to_scalar.hpp>

#include <node_cuda/utilities/error.hpp>
#include <node_cuda/utilities/napi_to_cpp.hpp>

#include <node_rmm/utilities/napi_to_cpp.hpp>

#include <nv_node/utilities/cpp_to_napi.hpp>

#include <cudf/column/column_view.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <napi.h>
#include <type_traits>

namespace nv {

Napi::Function Scalar::Init(Napi::Env const& env, Napi::Object exports) {
  return DefineClass(
    env,
    "Scalar",
    {
      InstanceAccessor("type", &Scalar::type, nullptr, napi_enumerable),
      InstanceAccessor("value", &Scalar::get_value, &Scalar::set_value, napi_enumerable),
    });
}

Scalar::wrapper_t Scalar::New(Napi::Env const& env, std::unique_ptr<cudf::scalar> scalar) {
  auto opts = Napi::Object::New(env);
  opts.Set("type", cudf_to_arrow_type(env, scalar->type()));
  auto wrap = Napi::External<std::unique_ptr<cudf::scalar>>::New(env, &scalar);
  return EnvLocalObjectWrap<Scalar>::New(env, {opts, wrap});
}

Scalar::wrapper_t Scalar::New(Napi::Env const& env, Napi::Number const& value) {
  return New(env, value, cudf::data_type{cudf::type_id::FLOAT64});
}

Scalar::wrapper_t Scalar::New(Napi::Env const& env, Napi::BigInt const& value) {
  return New(env, value, cudf::data_type{cudf::type_id::INT64});
}

Scalar::wrapper_t Scalar::New(Napi::Env const& env, Napi::String const& value) {
  return New(env, value, cudf::data_type{cudf::type_id::STRING});
}

Scalar::wrapper_t Scalar::New(Napi::Env const& env,
                              Napi::Value const& value,
                              cudf::data_type type) {
  auto opts = Napi::Object::New(env);
  opts.Set("value", value);
  opts.Set("type", cudf_to_arrow_type(env, type));
  return EnvLocalObjectWrap<Scalar>::New(env, {opts});
}

Scalar::Scalar(CallbackArgs const& args) : EnvLocalObjectWrap<Scalar>(args) {
  auto env = args.Env();

  NODE_CUDF_EXPECT(args.Length() > 0 && args[0].IsObject(),
                   "Scalar constructor expects an Object options argument",
                   env);

  NapiToCPP::Object props = args[0];

  NODE_CUDF_EXPECT(
    props.Has("type"), "Scalar constructor expects options to have a 'type' field", env);

  NODE_CUDF_EXPECT(
    props.Get("type").IsObject(), "Scalar constructor expects 'type' option to be a DataType", env);

  type_ = Napi::Persistent<Napi::Object>(props.Get("type"));

  if (args.Length() == 1) {
    *this = cudf::make_default_constructed_scalar(this->type());
    if (props.Has("value")) { set_value(args, props.Get("value")); }
  } else if (args.Length() == 2 && args[1].IsExternal()) {
    *this = std::move(*args[1].As<Napi::External<std::unique_ptr<cudf::scalar>>>().Data());
  }
}

Scalar::operator cudf::scalar&() const { return *scalar_; }

Napi::Value Scalar::type(Napi::CallbackInfo const& info) { return type_.Value(); }

Napi::Value Scalar::get_value() const { return Napi::Value::From(Env(), scalar_); }

Napi::Value Scalar::get_value(Napi::CallbackInfo const& info) { return this->get_value(); }

void Scalar::set_value(Napi::CallbackInfo const& info, Napi::Value const& value) {
  if (value.IsNull() or value.IsUndefined()) {
    this->set_valid(false);
  } else {
    cudf::type_dispatcher(this->type(), detail::set_scalar_value{value}, scalar_);
  }
}

}  // namespace nv
