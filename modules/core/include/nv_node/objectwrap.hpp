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

#include <nv_node/addon.hpp>
#include <nv_node/utilities/cpp_to_napi.hpp>

#include <cassert>

namespace nv {

template <typename T>
struct Wrapper : public Napi::Object {
  // Create a new _empty_ Wrapper<T> instance
  inline Wrapper() : Napi::Object() {}

  // Mirror the Napi::Object parent constructor
  inline Wrapper(napi_env env, napi_value value) : Napi::Object(env, value) {
    if (not T::IsInstance(*this)) {
      NAPI_THROW(Napi::Error::New(Env(), "Attempted to create Wrapper for incompatible Object."));
    }
    _parent = T::Unwrap(*this);
  }

  // Copy from an Napi::Object to Wrapper<T>
  inline Wrapper(Napi::Object const& value) : Napi::Object(value.Env(), value) {
    if (not T::IsInstance(*this)) {
      NAPI_THROW(Napi::Error::New(Env(), "Attempted to create Wrapper for incompatible Object."));
    }
    _parent = T::Unwrap(*this);
  }

  // use default move constructor
  inline Wrapper(Wrapper<T>&&) = default;

  // use default copy constructor
  inline Wrapper(Wrapper<T> const&) = default;

  // use default move assignment operator
  inline Wrapper<T>& operator=(Wrapper<T>&&) = default;

  // use default copy assignment operator
  inline Wrapper<T>& operator=(Wrapper<T> const&) = default;

  // Allow converting Wrapper<T> to T&
  inline operator T&() const noexcept { return *_parent; }

  // Access T members via pointer-access operator
  inline T* operator->() const noexcept { return _parent; }

  // Allow converting to T& via dereference operator
  inline T& operator*() const noexcept { return *_parent; }

 private:
  T* _parent{nullptr};
};

template <typename T>
struct EnvLocalObjectWrap : public Napi::ObjectWrap<T>,  //
                            public Napi::Reference<Wrapper<T>> {
  using wrapper_t = Wrapper<T>;

  using Napi::ObjectWrap<T>::Env;

  inline EnvLocalObjectWrap(Napi::CallbackInfo const& info) : Napi::ObjectWrap<T>(info){};

  /**
   * @brief Retrieve the JavaScript constructor function for C++ type `T`.
   *
   * @param env The currently active JavaScript environment.
   * @return Napi::Function The JavaScript constructor function for C++ type T.
   */
  inline static Napi::Function Constructor(Napi::Env env) {
    return env.GetInstanceData<EnvLocalAddon>()->GetConstructor<T>();
  }

  /**
   * @brief Check whether an Napi value is an instance of C++ type `T`.
   *
   * @param value The Napi::Value to test
   * @return true if the value is an instance of type T
   * @return false if the value is not an instance of type T
   */
  inline static bool IsInstance(Napi::Value const& value) {
    return value.IsObject() and value.As<Napi::Object>().InstanceOf(Constructor(value.Env()));
  }

  /**
   * @brief Construct a new instance of `T` from C++.
   *
   * @param env The currently active JavaScript environment.
   * @param args The Napi::Value arguments to pass to the constructor for type `T`.
   * @return Wrapper<T> The JavaScript Object representing the new C++ instance.
   * The lifetime of the C++ instance is tied to the lifetime of the returned wrapper.
   */
  inline static wrapper_t New(Napi::Env env, std::initializer_list<napi_value> const& args) {
    return env.GetInstanceData<EnvLocalAddon>()->GetConstructor<T>().New(args);
  }

  /**
   * @brief Construct a new instance of `T` from C++.
   *
   * @param env The currently active JavaScript environment.
   * @param args The Napi::Value arguments to pass to the constructor for type `T`.
   * @return Wrapper<T> The JavaScript Object representing the new C++ instance.
   * The lifetime of the C++ instance is tied to the lifetime of the returned wrapper.
   */
  inline static wrapper_t New(Napi::Env env, std::vector<napi_value> const& args) {
    return env.GetInstanceData<EnvLocalAddon>()->GetConstructor<T>().New(args);
  }

  /**
   * @brief Construct a new instance of `T` from C++ values.
   *
   * @param env The currently active JavaScript environment.
   * @param args The Napi::Value arguments to pass to the constructor for type `T`.
   * @return Wrapper<T> The JavaScript Object representing the new C++ instance.
   * The lifetime of the C++ instance is tied to the lifetime of the returned wrapper.
   */
  template <typename... Args>
  inline static wrapper_t New(Napi::Env env, Args&&... args) {
    std::vector<napi_value> napi_values;
    napi_values.reserve(sizeof...(Args));
    nv::casting::for_each(
      std::make_tuple<Args...>(std::forward<Args>(args)...),
      [&](auto const& x) mutable { napi_values.push_back(Napi::Value::From(env, x)); });
    return env.GetInstanceData<EnvLocalAddon>()->GetConstructor<T>().New(napi_values);
  }

  inline operator wrapper_t() const { return Value(); }

  // inline operator Napi::Value() const { return Value(); }

  // inline operator Napi::Object() const { return Value(); }

  inline wrapper_t Value() const { return Napi::ObjectWrap<T>::Value(); }
};

}  // namespace nv

namespace Napi {
namespace details {

// Tell `Napi::Value::From` that `Wrapper<T>` instances cannot be converted to Strings
template <typename T>
struct can_make_string<nv::Wrapper<T>> : std::false_type {};

// Tell `Napi::Value::From` that `EnvLocalObjectWrap<T>` instances cannot be converted to Strings
template <typename T>
struct can_make_string<nv::EnvLocalObjectWrap<T>> : std::false_type {};

template <typename T>
struct vf_fallback<nv::Wrapper<T>> {
  inline static Value From(napi_env env, nv::Wrapper<T> const& wrapper) { return wrapper; }
};

}  // namespace details
}  // namespace Napi
