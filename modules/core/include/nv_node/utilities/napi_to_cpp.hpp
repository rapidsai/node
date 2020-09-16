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

#include "nv_node/utilities/span.hpp"

// #include <cuda.h>
// #include <cuda_runtime.h>
#include <napi.h>

#include <cstring>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace nv {

struct NapiToCPP {
  Napi::Value const val;
  inline NapiToCPP(const Napi::Value& val) : val(val) {}

  inline std::ostream& operator<<(std::ostream& os) const {
    return os << val.ToString().operator std::string();
  }

  //
  // Napi identities
  //
  inline operator Napi::Value() const { return val; }
  inline operator Napi::Boolean() const { return val.As<Napi::Boolean>(); }
  inline operator Napi::Number() const { return val.As<Napi::Number>(); }
  inline operator Napi::String() const { return val.As<Napi::String>(); }
  inline operator Napi::Object() const { return val.As<Napi::Object>(); }
  inline operator Napi::Array() const { return val.As<Napi::Array>(); }
  inline operator Napi::Function() const { return val.As<Napi::Function>(); }
  inline operator Napi::Error() const { return val.As<Napi::Error>(); }
  inline operator Napi::ArrayBuffer() const {
    if (val.IsArrayBuffer()) { return val.As<Napi::ArrayBuffer>(); }
    if (val.IsDataView()) { return val.As<Napi::DataView>().ArrayBuffer(); }
    if (val.IsTypedArray()) { return val.As<Napi::TypedArray>().ArrayBuffer(); }
    auto msg = "Value must be ArrayBuffer or ArrayBufferView";
    NAPI_THROW(Napi::Error::New(val.Env(), msg), val.Env().Undefined());
  }
  inline operator Napi::DataView() const { return val.As<Napi::DataView>(); }
  inline operator Napi::TypedArray() const { return val.As<Napi::TypedArray>(); }
  template <typename T>
  inline operator Napi::Buffer<T>() const {
    return val.As<Napi::Buffer<T>>();
  }

  //
  // Primitives
  //
  inline operator bool() const { return val.ToBoolean(); }
  inline operator float() const { return to_numeric<float>(); }
  inline operator double() const { return to_numeric<double>(); }
  inline operator int8_t() const { return to_numeric<int64_t>(); }
  inline operator int16_t() const { return to_numeric<int64_t>(); }
  inline operator int32_t() const { return to_numeric<int32_t>(); }
  inline operator int64_t() const { return to_numeric<int64_t>(); }
  inline operator uint8_t() const { return to_numeric<int64_t>(); }
  inline operator uint16_t() const { return to_numeric<int64_t>(); }
  inline operator uint32_t() const { return to_numeric<uint32_t>(); }
  inline operator uint64_t() const { return to_numeric<int64_t>(); }
  inline operator std::string() const { return val.ToString(); }
  inline operator std::u16string() const { return val.ToString(); }
  inline operator napi_value() const { return val.operator napi_value(); }

  //
  // Arrays
  //
  template <typename T>
  inline operator std::vector<T>() const {
    if (val.IsArray()) {
      std::vector<T> vec{};
      auto arr = val.As<Napi::Array>();
      for (uint32_t i = 0; i < arr.Length(); ++i) {
        vec.push_back(NapiToCPP(arr.Get(i)).operator T());
      }
      return vec;
    }
    if (!(val.IsNull() || val.IsEmpty())) {  //
      return std::vector<T>{this->operator T()};
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
        Key k  = NapiToCPP(keys.Get(i));
        Val v  = NapiToCPP(obj.Get(keys.Get(i)));
        map[k] = v;
      }
      return map;
    }
    return std::map<Key, Val>{};
  }

  //
  // Pointers
  //
  inline operator void*() const {  //
    return static_cast<void*>(as_span<char>().data());
  }
  inline operator void const *() const {  //
    return static_cast<void const*>(as_span<char>().data());
  }
  template <typename T>
  inline operator T*() const {
    return as_span<T>();
  }
  template <typename T>
  inline operator Span<T>() const {
    return as_span<T>();
  }

#ifdef CUDA_VERSION
  //
  // CUDA Driver type conversion helpers
  //
  inline operator CUresult() const { return static_cast<CUresult>(this->operator int64_t()); }
  inline operator CUdevice_attribute() const {
    return static_cast<CUdevice_attribute>(this->operator int64_t());
  }
  inline operator CUpointer_attribute() const {
    return static_cast<CUpointer_attribute>(this->operator int64_t());
  }
#endif

#ifdef CUDART_VERSION
  //
  // CUDA Runtime type conversion helpers
  //
  inline operator cudaArray_t() const {
    return reinterpret_cast<cudaArray_t>(this->operator char*());
  }
  inline operator cudaGraphicsResource_t() const {
    return reinterpret_cast<cudaGraphicsResource_t>(this->operator char*());
  }
  inline operator cudaIpcMemHandle_t*() const {
    return reinterpret_cast<cudaIpcMemHandle_t*>(this->operator char*());
  }
  inline operator cudaStream_t() const {
    return reinterpret_cast<cudaStream_t>(this->operator char*());
  };
  inline operator cudaUUID_t*() const {
    return reinterpret_cast<cudaUUID_t*>(this->operator char*());
  }
#endif

#ifdef GLEW_VERSION
  inline operator GLsync() const { return reinterpret_cast<GLsync>(this->operator char*()); }
#endif

 protected:
  template <typename T>
  inline Span<T> as_span() const {
    // Easy cases -- ArrayBuffers and ArrayBufferViews
    if (val.IsDataView()) { return Span<T>(val.As<Napi::DataView>()); }
    if (val.IsTypedArray()) { return Span<T>(val.As<Napi::TypedArray>()); }
    if (val.IsArrayBuffer()) { return Span<T>(val.As<Napi::ArrayBuffer>()); }

    // Value could be an Napi::External wrapper around a raw pointer
    if (val.IsExternal()) { return Span<T>(val.As<Napi::External<T>>()); }

    // Allow treating JS numbers as memory addresses. Useful, but dangerous.
    if (val.IsNumber()) {  //
      return Span<T>(reinterpret_cast<char*>(val.ToNumber().Int64Value()), 0);
    }

    // Objects wrapping raw memory with some conventions:
    // * Objects with a numeric "ptr" field and optional numeric "byteLength" field
    // * Objects with a "buffer" field, which itself has numeric "byteLength" and "ptr" fields
    // If the wrapping object has a numeric "byteOffset" field, it is propagated to the span.
    if (val.IsObject() and not val.IsNull()) {
      size_t length{0};
      size_t offset{0};
      auto obj = val.As<Napi::Object>();
      if (obj.Has("byteOffset") and obj.Get("byteOffset").IsNumber()) {
        offset = NapiToCPP(obj.Get("byteOffset"));
      }
      if (obj.Has("byteLength") and obj.Get("byteLength").IsNumber()) {
        length = NapiToCPP(obj.Get("byteLength"));
      }
      if (obj.Has("buffer") and obj.Get("buffer").IsObject()) {
        obj = obj.Get("buffer").As<Napi::Object>();
      }
      if (obj.Has("ptr")) {
        if (obj.Get("ptr").IsNumber()) {
          return Span<T>(static_cast<char*>(NapiToCPP(obj.Get("ptr"))) + offset, length);
        }
        NAPI_THROW("Expected `ptr` to be numeric");
      }
    }
    return Span<T>(static_cast<char*>(nullptr), 0);
  }

  template <typename T>
  inline T to_numeric() const {
    if (val.IsNull() || val.IsEmpty()) { return 0; }
    if (val.IsNumber() || val.IsString()) { return val.ToNumber(); }
    if (val.IsBoolean()) { return val.ToBoolean().operator bool(); }

    // Accept single-element numeric Arrays (e.g. OpenGL)
    if (val.IsArray()) {
      auto ary = val.As<Napi::Array>();
      if (ary.Length() == 1) {
        auto elt = ary.Get(uint32_t{0});
        if (elt.IsNumber() || elt.IsBigInt()) {
          return static_cast<T>(NapiToCPP(ary.Get(uint32_t{0})));
        }
        NAPI_THROW("Expected `0` to be numeric");
      }
      return 0;
    }
    // Accept Objects with a numeric "ptr" field (e.g. OpenGL)
    if (val.IsObject()) {
      auto obj = val.As<Napi::Object>();
      if (obj.Has("ptr")) {
        auto ptr = obj.Get("ptr");
        if (ptr.IsNumber()) {  //
          return static_cast<T>(NapiToCPP(ptr));
        }
        NAPI_THROW("Expected `ptr` to be numeric");
      }
    }
    return 0;
  }
};

}  // namespace nv
