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

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <napi.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <map>
#include <string>
#include <type_traits>

namespace node_nvencoder {

struct FromJS {
  Napi::Value val;
  inline FromJS(const Napi::Value& val) : val(val) {}

  //
  // Napi identities
  //
  inline operator Napi::Boolean() const { return val.ToBoolean(); }
  inline operator Napi::Number() const { return val.ToNumber(); }
  inline operator Napi::String() const { return val.ToString(); }
  inline operator Napi::Object() const { return val.ToObject(); }
  inline operator Napi::Array() const { return val.As<Napi::Array>(); }
  inline operator Napi::Function() const { return val.As<Napi::Function>(); }
  inline operator Napi::Error() const { return val.As<Napi::Error>(); }
  inline operator Napi::TypedArray() const { return val.As<Napi::TypedArray>(); }
  inline operator Napi::ArrayBuffer() const {
    if (val.IsArrayBuffer()) { return val.As<Napi::ArrayBuffer>(); }
    if (val.IsDataView()) { return val.As<Napi::DataView>().ArrayBuffer(); }
    if (val.IsTypedArray()) { return val.As<Napi::TypedArray>().ArrayBuffer(); }
    auto msg = "Value must be ArrayBuffer or ArrayBufferView";
    NAPI_THROW(Napi::Error::New(val.Env(), msg), val.Env().Undefined());
  }

  //
  // Primitives
  //
  inline operator napi_value() const { return val.operator napi_value(); }
  inline operator bool() const { return val.ToBoolean().operator bool(); }
  inline operator float() const { return static_cast<float>(val.ToNumber().operator float()); }
  inline operator double() const { return static_cast<double>(val.ToNumber().operator double()); }
  inline operator int8_t() const { return static_cast<int8_t>(val.ToNumber().operator int64_t()); }
  inline operator int16_t() const {
    return static_cast<int16_t>(val.ToNumber().operator int64_t());
  }
  inline operator int32_t() const {
    return static_cast<int32_t>(val.ToNumber().operator int32_t());
  }
  inline operator int64_t() const {
    return static_cast<int64_t>(val.ToNumber().operator int64_t());
  }
  inline operator uint8_t() const {
    return static_cast<uint8_t>(val.ToNumber().operator int64_t());
  }
  inline operator uint16_t() const {
    return static_cast<uint16_t>(val.ToNumber().operator int64_t());
  }
  inline operator uint32_t() const {
    return static_cast<uint32_t>(val.ToNumber().operator uint32_t());
  }
  inline operator uint64_t() const {
    return static_cast<uint64_t>(val.ToNumber().operator int64_t());
  }
  inline operator std::string() const { return val.ToString().operator std::string(); }
  inline operator std::u16string() const { return val.ToString().operator std::u16string(); }
  inline operator char*() const {
    std::string str = FromJS(val);
    auto ptr        = reinterpret_cast<char*>(malloc(str.size()));
    memcpy(ptr, str.c_str(), str.size());
    return ptr;
  }

  inline operator CUresult() const {
    return static_cast<CUresult>(this->val.ToNumber().Int32Value());
  }
  inline operator CUstream() const {
    return reinterpret_cast<CUstream>(this->val.ToNumber().Int64Value());
  }
  inline operator CUdevice_attribute() const {
    return static_cast<CUdevice_attribute>(this->val.ToNumber().Uint32Value());
  }
  inline operator CUpointer_attribute() const {
    return static_cast<CUpointer_attribute>(this->val.ToNumber().Uint32Value());
  }

  //
  // Arrays
  //
  template <typename T>
  inline operator std::vector<T>() const {
    if (val.IsArray()) {
      std::vector<T> vec{};
      auto arr = val.As<Napi::Array>();
      for (uint32_t i = 0; i < arr.Length(); ++i) {
        vec.push_back(FromJS(arr.Get(i)).operator T());
      }
      return vec;
    }
    return std::vector<T>{};
  }

  //
  // Objects
  //

  template <typename Key, typename Val>
  inline operator std::map<Key, Val>() const {
    if (val.IsObject()) {
      std::map<Key, Val> map{};
      auto obj  = val.As<Napi::Object>();
      auto keys = obj.GetPropertyNames();
      for (uint32_t i = 0; i < keys.Length(); ++i) {
        Key k  = FromJS(keys.Get(i));
        Val v  = FromJS(obj.Get(keys.Get(i)));
        map[k] = v;
      }
      return map;
    }
    return std::map<Key, Val>{};
  }

  //
  // Pointers
  //
  inline operator void*() const {
    if (val.IsExternal()) { return val.As<Napi::External<void>>().Data(); }
    if (val.IsArrayBuffer()) { return val.As<Napi::ArrayBuffer>().Data(); }
    if (val.IsDataView()) {
      auto offset = val.As<Napi::DataView>().ByteOffset();
      auto buffer = val.As<Napi::DataView>().ArrayBuffer();
      return reinterpret_cast<uint8_t*>(buffer.Data()) + offset;
    }
    if (val.IsTypedArray()) {
      auto offset = val.As<Napi::TypedArray>().ByteOffset();
      auto buffer = val.As<Napi::TypedArray>().ArrayBuffer();
      return reinterpret_cast<uint8_t*>(buffer.Data()) + offset;
    }
    if (val.IsObject()) {
      auto obj = val.As<Napi::Object>();
      if (obj.Has("buffer")) { obj = obj.Get("buffer").As<Napi::Object>(); }
      if (obj.Has("byteLength")) {
        if (obj.Has("ptr") && obj.Get("ptr").IsNumber()) {
          return reinterpret_cast<void*>(obj.Get("ptr").As<Napi::Number>().Int64Value());
        }
        NAPI_THROW("Expected object with a `ptr` field");
      }
    }
    return reinterpret_cast<void*>(val.operator napi_value());
  }

  inline operator cudaUUID_t() const {
    return *reinterpret_cast<cudaUUID_t*>(this->operator void*());
  }
  inline operator CUcontext() const { return reinterpret_cast<CUcontext>(this->operator void*()); }
  inline operator CUfunction() const {
    return reinterpret_cast<CUfunction>(this->operator void*());
  }
  inline operator CUdeviceptr() const {
    return reinterpret_cast<CUdeviceptr>(this->operator void*());
  }
  inline operator CUipcMemHandle() const {
    return *reinterpret_cast<CUipcMemHandle*>(this->operator void*());
  }
  inline operator cudaGraphicsResource_t() const {
    return reinterpret_cast<cudaGraphicsResource_t>(this->operator void*());
  }
  inline operator cudaIpcMemHandle_t() const {
    return *reinterpret_cast<cudaIpcMemHandle_t*>(this->operator void*());
  }

#define POINTER_CONVERSION_OPERATOR(T)                                     \
  inline operator T*() const {                                             \
    if (val.IsArray()) {                                                   \
      std::vector<T> vec = FromJS(val);                                    \
      auto len           = vec.size() * sizeof(T);                         \
      auto ptr           = std::malloc(vec.size() * sizeof(T));            \
      auto ary           = reinterpret_cast<T*>(ptr);                      \
      for (int32_t i = 0; i < vec.size(); ++i) { *(ary + i) = vec.at(i); } \
      return ary;                                                          \
    }                                                                      \
    return reinterpret_cast<T*>(this->operator void*());                   \
  }                                                                        \
  inline operator std::pair<size_t, T*>() const {                          \
    if (val.IsArray()) {                                                   \
      auto ptr = this->operator T*();                                      \
      auto ary = val.As<Napi::Array>();                                    \
      return std::make_pair(ary.Length(), ptr);                            \
    }                                                                      \
    if (val.IsArrayBuffer()) {                                             \
      auto ary = val.As<Napi::ArrayBuffer>();                              \
      auto len = ary.ByteLength() / sizeof(T);                             \
      return std::make_pair(len, this->operator T*());                     \
    }                                                                      \
    if (val.IsDataView()) {                                                \
      auto ary = val.As<Napi::DataView>();                                 \
      auto len = ary.ByteLength() / sizeof(T);                             \
      return std::make_pair(len, this->operator T*());                     \
    }                                                                      \
    if (val.IsTypedArray()) {                                              \
      auto ary = val.As<Napi::TypedArray>();                               \
      auto len = ary.ByteLength() / sizeof(T);                             \
      return std::make_pair(len, this->operator T*());                     \
    }                                                                      \
    return std::make_pair(size_t{0}, nullptr);                             \
  }

  POINTER_CONVERSION_OPERATOR(long)
  POINTER_CONVERSION_OPERATOR(float)
  POINTER_CONVERSION_OPERATOR(double)
  POINTER_CONVERSION_OPERATOR(int8_t)
  POINTER_CONVERSION_OPERATOR(int16_t)
  POINTER_CONVERSION_OPERATOR(int32_t)
  POINTER_CONVERSION_OPERATOR(uint8_t)
  POINTER_CONVERSION_OPERATOR(uint16_t)
  POINTER_CONVERSION_OPERATOR(uint32_t)

#undef POINTER_CONVERSION_OPERATOR
};

struct ToNapi {
  Napi::Env env;
  inline ToNapi(Napi::Env const& env) : env(env) {}

  // Primitives
  Napi::Boolean inline operator()(const bool& val) const {
    return Napi::Boolean::New(this->env, val);
  }
  Napi::Number inline operator()(const float& val) const {
    return Napi::Number::New(this->env, val);
  }
  Napi::Number inline operator()(const double& val) const {
    return Napi::Number::New(this->env, val);
  }
  Napi::Number inline operator()(const int8_t& val) const {
    return Napi::Number::New(this->env, val);
  }
  Napi::Number inline operator()(const int16_t& val) const {
    return Napi::Number::New(this->env, val);
  }
  Napi::Number inline operator()(const int32_t& val) const {
    return Napi::Number::New(this->env, val);
  }
  Napi::Number inline operator()(const int64_t& val) const {
    return Napi::Number::New(this->env, val);
  }
  Napi::Number inline operator()(const uint8_t& val) const {
    return Napi::Number::New(this->env, val);
  }
  Napi::Number inline operator()(const uint16_t& val) const {
    return Napi::Number::New(this->env, val);
  }
  Napi::Number inline operator()(const uint32_t& val) const {
    return Napi::Number::New(this->env, val);
  }
  Napi::Number inline operator()(const uint64_t& val) const {
    return Napi::Number::New(this->env, val);
  }
  Napi::Value inline operator()(CUstream val) const {
    return Napi::Number::New(this->env, reinterpret_cast<int64_t>(val));
  }
  Napi::String inline operator()(const char* val) const {
    return Napi::String::New(this->env, val);
  }
  Napi::String inline operator()(const std::string& val) const {
    return Napi::String::New(this->env, val);
  }
  Napi::String inline operator()(const std::u16string& val) const {
    return Napi::String::New(this->env, val);
  }

  //
  // Arrays
  //

  template <int N>
  Napi::Array inline operator()(const int (&arr)[N]) const {
    return (*this)(std::vector<int>{arr, arr + N});
  }

  template <typename T>
  Napi::Array inline operator()(const std::vector<T>& vec) const {
    uint32_t idx = 0;
    auto arr     = Napi::Array::New(this->env, vec.size());
    std::for_each(vec.begin(), vec.end(), [&idx, &cast_t = *this, &arr](const T& val) {
      arr.Set(cast_t(idx++), cast_t(val));
    });
    return arr;
  }

  //
  // Objects
  //

  template <typename Key, typename Val>
  Napi::Object inline operator()(const std::map<Key, Val> map) const {
    auto cast_t = *this;
    auto obj    = Napi::Object::New(this->env);
    for (auto pair : map) { obj.Set(cast_t(pair.first), cast_t(pair.second)); }
    return obj;
  }

  template <typename T>
  Napi::Object inline operator()(const std::vector<T>& vals,
                                 const std::vector<std::string>& keys) const {
    auto cast_t = *this;
    auto val    = vals.begin();
    auto key    = keys.begin();
    auto obj    = Napi::Object::New(this->env);
    while ((val != vals.end()) && (key != keys.end())) {
      obj.Set(cast_t(*key), cast_t(*val));
      std::advance(key, 1);
      std::advance(val, 1);
    }
    return obj;
  }

  //
  // Pointers
  //

  template <typename Finalizer>
  Napi::ArrayBuffer inline operator()(void* ptr, size_t size, Finalizer finalizer) const {
    return Napi::ArrayBuffer::New(this->env, ptr, size, finalizer);
  }
  Napi::ArrayBuffer inline operator()(void* ptr, size_t size) const {
    return Napi::ArrayBuffer::New(this->env, ptr, size);
  }
  Napi::Uint8Array inline operator()(const cudaUUID_t& val) const {
    auto arr = Napi::Uint8Array::New(this->env, sizeof(cudaUUID_t));
    memcpy(arr.ArrayBuffer().Data(), &val.bytes, sizeof(cudaUUID_t));
    return arr;
  }
  Napi::Uint8Array inline operator()(const CUipcMemHandle& ptr) const {
    auto arr = Napi::Uint8Array::New(this->env, CU_IPC_HANDLE_SIZE);
    memcpy(arr.ArrayBuffer().Data(), &ptr.reserved, CU_IPC_HANDLE_SIZE);
    return arr;
  }
  Napi::Uint8Array inline operator()(const cudaIpcMemHandle_t& ptr) const {
    auto arr = Napi::Uint8Array::New(this->env, CU_IPC_HANDLE_SIZE);
    memcpy(arr.ArrayBuffer().Data(), &ptr.reserved, CU_IPC_HANDLE_SIZE);
    return arr;
  }
  Napi::Value inline operator()(CUcontext val) const {
    return (val == NULL) ? this->env.Null() : Napi::External<void>::New(this->env, val);
  }
  Napi::Value inline operator()(CUfunction val) const {
    return (val == NULL) ? this->env.Null() : Napi::External<void>::New(this->env, val);
  }
  Napi::Value inline operator()(CUgraphicsResource val) const {
    return (val == NULL) ? this->env.Null() : Napi::External<void>::New(this->env, val);
  }
  Napi::Value inline operator()(cudaGraphicsResource_t val) const {
    return (val == NULL) ? this->env.Null() : Napi::External<void>::New(this->env, val);
  }
};
}  // namespace node_nvencoder
