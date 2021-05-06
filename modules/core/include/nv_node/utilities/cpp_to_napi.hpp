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

#pragma once

#include "span.hpp"

#include <napi.h>

#include <initializer_list>
#include <map>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace nv {

namespace casting {

template <std::size_t I = 0, typename FuncT, typename... Tp>
inline typename std::enable_if<I == sizeof...(Tp), void>::type for_each(
  std::tuple<Tp...> const&,
  FuncT)  // Unused arguments are given no names.
{}

template <std::size_t I = 0, typename FuncT, typename... Tp>
  inline typename std::enable_if <
  I<sizeof...(Tp), void>::type for_each(std::tuple<Tp...> const& t, FuncT f) {
  f(std::get<I>(t));
  for_each<I + 1, FuncT, Tp...>(t, f);
}

}  // namespace casting

struct CPPToNapi {
  inline CPPToNapi(Napi::Env const& env) : _env(env) {}
  inline CPPToNapi(Napi::CallbackInfo const& info) : CPPToNapi(info.Env()) {}

  inline Napi::Env Env() const { return _env; }

  template <typename Arg>
  Napi::Value operator()(Arg const&) const;

  //
  // Napi identities
  //
  inline napi_value operator()(napi_value const& val) const { return val; }
  inline Napi::Value operator()(Napi::Value const& val) const { return val; }
  inline Napi::Boolean operator()(Napi::Boolean const& val) const { return val; }
  inline Napi::Number operator()(Napi::Number const& val) const { return val; }
  inline Napi::String operator()(Napi::String const& val) const { return val; }
  inline Napi::Object operator()(Napi::Object const& val) const { return val; }
  inline Napi::Array operator()(Napi::Array const& val) const { return val; }
  inline Napi::Function operator()(Napi::Function const& val) const { return val; }
  inline Napi::Error operator()(Napi::Error const& val) const { return val; }
  inline Napi::ArrayBuffer operator()(Napi::ArrayBuffer const& val) const { return val; }
  inline Napi::DataView operator()(Napi::DataView const& val) const { return val; }
  inline Napi::TypedArray operator()(Napi::TypedArray const& val) const { return val; }
  template <typename T>
  inline Napi::TypedArrayOf<T> operator()(Napi::TypedArrayOf<T> const& val) const {
    return val;
  }
  template <typename T>
  inline Napi::Buffer<T> operator()(Napi::Buffer<T> const& val) const {
    return val;
  }

  // Primitives
  inline Napi::Boolean operator()(bool const& val) const { return Napi::Boolean::New(Env(), val); }
  inline Napi::Number operator()(float const& val) const { return Napi::Number::New(Env(), val); }
  inline Napi::Number operator()(double const& val) const { return Napi::Number::New(Env(), val); }
  inline Napi::Number operator()(int8_t const& val) const { return Napi::Number::New(Env(), val); }
  inline Napi::Number operator()(int16_t const& val) const { return Napi::Number::New(Env(), val); }
  inline Napi::Number operator()(int32_t const& val) const { return Napi::Number::New(Env(), val); }
  inline Napi::Number operator()(int64_t const& val) const { return Napi::Number::New(Env(), val); }
  inline Napi::Number operator()(uint8_t const& val) const { return Napi::Number::New(Env(), val); }
  inline Napi::Number operator()(uint16_t const& val) const {
    return Napi::Number::New(Env(), val);
  }
  inline Napi::Number operator()(uint32_t const& val) const {
    return Napi::Number::New(Env(), val);
  }
  inline Napi::Number operator()(uint64_t const& val) const {
    return Napi::Number::New(Env(), val);
  }
  inline Napi::String operator()(std::string const& val) const {
    return Napi::String::New(Env(), val);
  }
  inline Napi::String operator()(std::u16string const& val) const {
    return Napi::String::New(Env(), val);
  }

  //
  // Pair
  //
  template <typename T>
  Napi::Object inline operator()(std::pair<T, T> const& pair) const {
    auto cast_t = *this;
    auto obj    = Napi::Array::New(Env(), 2);
    obj.Set(uint32_t(0), cast_t(pair.first));
    obj.Set(uint32_t(1), cast_t(pair.second));
    return obj;
  }

  //
  // Arrays
  //
  // template <typename T, int N>
  // inline Napi::Array operator()(const T (&arr)[N]) const {
  //   return (*this)(std::vector<T>{arr, arr + N});
  // }

  template <typename T>
  inline Napi::Array operator()(std::vector<T> const& vec) const {
    uint32_t idx = 0;
    auto arr     = Napi::Array::New(Env(), vec.size());
    std::for_each(
      vec.begin(), vec.end(), [&](T const& val) mutable { arr.Set((*this)(idx++), (*this)(val)); });
    return arr;
  }

  template <typename T>
  inline Napi::Array operator()(std::initializer_list<T> const& vec) const {
    uint32_t idx = 0;
    auto arr     = Napi::Array::New(Env(), vec.size());
    std::for_each(
      vec.begin(), vec.end(), [&](T const& val) mutable { arr.Set((*this)(idx++), (*this)(val)); });
    return arr;
  }

  //
  // Objects
  //
  template <typename Key, typename Val>
  Napi::Object inline operator()(const std::map<Key, Val> map) const {
    auto cast_t = *this;
    auto obj    = Napi::Object::New(Env());
    for (auto pair : map) { obj.Set(cast_t(pair.first), cast_t(pair.second)); }
    return obj;
  }

  template <typename T>
  inline Napi::Object operator()(std::vector<T> const& vals,
                                 std::vector<std::string> const& keys) const {
    auto self = *this;
    auto val  = vals.begin();
    auto key  = keys.begin();
    auto obj  = Napi::Object::New(Env());
    while ((val != vals.end()) && (key != keys.end())) {
      obj.Set(self(*key), self(*val));
      std::advance(key, 1);
      std::advance(val, 1);
    }
    return obj;
  }

  template <typename... Vals>
  Napi::Object inline operator()(std::initializer_list<std::string> const& keys,
                                 std::tuple<Vals...> const& vals) const {
    auto cast_t = *this;
    auto key    = keys.begin();
    auto obj    = Napi::Object::New(Env());
    nv::casting::for_each(vals, [&](auto val) {
      obj.Set(cast_t(*key), cast_t(val));
      std::advance(key, 1);
    });
    return obj;
  }

  //
  // Pointers
  //

  template <typename T>
  inline Napi::Value operator()(T* data) const {
    return Napi::Number::New(Env(), reinterpret_cast<uintptr_t>(data));
  }

  template <typename T>
  inline Napi::Value operator()(T const* data) const {
    return Napi::Number::New(Env(), reinterpret_cast<uintptr_t>(data));
  }

  template <typename T>
  inline Napi::Value operator()(std::tuple<T const*, size_t> pair) const {
    auto size = sizeof(T) * std::get<1>(pair);
    auto buf  = Napi::ArrayBuffer::New(Env(), size);
    std::memcpy(buf.Data(), std::get<0>(pair), size);
    return buffer_to_typed_array<T>(buf);
  }

  template <typename T>
  inline Napi::Value operator()(Span<T> const& span) const {
    auto obj          = Napi::Object::New(Env());
    obj["ptr"]        = span.addr();
    obj["byteLength"] = span.size();
    return obj;
  }

#ifdef GLEW_VERSION
  inline Napi::External<void> operator()(GLsync const& sync) const {
    return Napi::External<void>::New(Env(), sync);
  }
#endif

#ifdef GLFW_APIENTRY_DEFINED
  inline Napi::Number operator()(GLFWcursor* ptr) const {
    return Napi::Number::New(Env(), reinterpret_cast<size_t>(ptr));
  }
  inline Napi::Number operator()(GLFWwindow* ptr) const {
    return Napi::Number::New(Env(), reinterpret_cast<size_t>(ptr));
  }
  inline Napi::Number operator()(GLFWmonitor* ptr) const {
    return Napi::Number::New(Env(), reinterpret_cast<size_t>(ptr));
  }
  inline Napi::Number operator()(GLFWglproc* ptr) const {
    return Napi::Number::New(Env(), reinterpret_cast<size_t>(ptr));
  }
#endif

 protected:
  template <typename T>
  Napi::Value buffer_to_typed_array(Napi::ArrayBuffer const& buf) const {
    auto len = const_cast<Napi::ArrayBuffer&>(buf).ByteLength() / sizeof(T);
    if (std::is_same<T, int8_t>() || std::is_same<T, char>()) {
      return Napi::Int8Array::New(Env(), len, buf, 0);
    }
    if (std::is_same<T, uint8_t>() || std::is_same<T, unsigned char>()) {
      return Napi::Uint8Array::New(Env(), len, buf, 0);
    }
    if (std::is_same<T, int16_t>() || std::is_same<T, short>()) {
      return Napi::Int16Array::New(Env(), len, buf, 0);
    }
    if (std::is_same<T, uint16_t>() || std::is_same<T, unsigned short>()) {
      return Napi::Uint16Array::New(Env(), len, buf, 0);
    }
    if (std::is_same<T, int32_t>()) {  //
      return Napi::Int32Array::New(Env(), len, buf, 0);
    }
    if (std::is_same<T, uint32_t>()) {  //
      return Napi::Uint32Array::New(Env(), len, buf, 0);
    }
    if (std::is_same<T, float>()) {  //
      return Napi::Float32Array::New(Env(), len, buf, 0);
    }
    if (std::is_same<T, double>()) {  //
      return Napi::Float64Array::New(Env(), len, buf, 0);
    }
    NAPI_THROW(std::runtime_error{"Unknown TypedArray type"}, env.Undefined());
  }

 private:
  Napi::Env const _env;
};

}  // namespace nv

namespace Napi {
namespace details {

template <typename T>
struct can_make_string<nv::Span<T>> : std::false_type {};

template <typename T>
struct vf_fallback<nv::Span<T>> {
  inline static Value From(napi_env env, nv::Span<T> const& span) {
    auto obj          = Napi::Object::New(env);
    obj["ptr"]        = span.addr();
    obj["byteLength"] = span.size();
    return obj;
  }
};

template <typename T>
struct vf_fallback<std::pair<T, T>> {
  inline static Value From(napi_env env, std::pair<T, T> const& pair) {
    auto obj = Array::New(env, 2);
    obj.Set(0u, Value::From(env, pair.first));
    obj.Set(1u, Value::From(env, pair.second));
    return obj;
  }
};

}  // namespace details
}  // namespace Napi
