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

#include "../objectwrap.hpp"
#include "span.hpp"

#include <napi.h>

#include <cstring>
#include <iostream>
#include <map>
#include <string>
#include <type_traits>
#include <vector>

namespace nv {

struct NapiToCPP {
  Napi::Value val;
  inline NapiToCPP(const Napi::Value& val) : val(val) {}

  struct Object {
    Napi::Object val;
    inline Object(Napi::Object const& o) : val(o) {}
    inline Object(Napi::Value const& o) : val(o.As<Napi::Object>()) {}
    inline operator Napi::Object() const { return val; }
    inline Napi::Env Env() const { return val.Env(); }
    inline bool Has(napi_value key) const { return val.Has(key); }
    inline bool Has(Napi::Value key) const { return val.Has(key); }
    inline bool Has(const char* key) const { return val.Has(key); }
    inline bool Has(const std::string& key) const { return val.Has(key); }
    inline NapiToCPP Get(napi_value key) const { return GetOrDefault(key, Env().Undefined()); }
    inline NapiToCPP Get(Napi::Value key) const { return GetOrDefault(key, Env().Undefined()); }
    inline NapiToCPP Get(const char* key) const { return GetOrDefault(key, Env().Undefined()); }
    inline NapiToCPP Get(std::string const& key) const {
      return GetOrDefault(key, Env().Undefined());
    }
    inline NapiToCPP GetOrDefault(napi_value key, Napi::Value const& default_val) const {
      return Has(key) ? val.Get(key) : default_val;
    }
    inline NapiToCPP GetOrDefault(Napi::Value key, Napi::Value const& default_val) const {
      return Has(key) ? val.Get(key) : default_val;
    }
    inline NapiToCPP GetOrDefault(const char* key, Napi::Value const& default_val) const {
      return Has(key) ? val.Get(key) : default_val;
    }
    inline NapiToCPP GetOrDefault(std::string const& key, Napi::Value const& default_val) const {
      return Has(key) ? val.Get(key) : default_val;
    }
  };

  inline std::ostream& operator<<(std::ostream& os) const {
    return os << val.ToString().operator std::string();
  }

  inline Napi::Env Env() const { return val.Env(); }
  inline bool IsEmpty() const { return val.IsEmpty(); }
  inline bool IsUndefined() const { return val.IsUndefined(); }
  inline bool IsNull() const { return val.IsNull(); }
  inline bool IsBoolean() const { return val.IsBoolean(); }
  inline bool IsNumber() const { return val.IsNumber(); }
  inline bool IsBigInt() const { return val.IsBigInt(); }
  inline bool IsDate() const { return val.IsDate(); }
  inline bool IsString() const { return val.IsString(); }
  inline bool IsSymbol() const { return val.IsSymbol(); }
  inline bool IsArray() const { return val.IsArray(); }
  inline bool IsArrayBuffer() const { return val.IsArrayBuffer(); }
  inline bool IsTypedArray() const { return val.IsTypedArray(); }
  inline bool IsObject() const { return val.IsObject(); }
  inline bool IsFunction() const { return val.IsFunction(); }
  inline bool IsPromise() const { return val.IsPromise(); }
  inline bool IsDataView() const { return val.IsDataView(); }
  inline bool IsBuffer() const { return val.IsBuffer(); }
  inline bool IsExternal() const { return val.IsExternal(); }

  inline bool IsMemoryViewLike() const {
    if (IsTypedArray() || IsDataView() || IsBuffer()) { return true; }
    if (val.IsObject() and not val.IsNull() and val.As<Napi::Object>().Has("buffer")) {
      return NapiToCPP(val.As<Napi::Object>().Get("buffer")).IsMemoryLike();
    }
    return false;
  }

  inline bool IsMemoryLike() const {
    if (IsArrayBuffer() || IsTypedArray() || IsDataView() || IsBuffer()) { return true; }
    if (val.IsObject() and not val.IsNull()) {
      auto obj = val.As<Napi::Object>();
      if (obj.Has("buffer")) {  //
        return NapiToCPP(obj.Get("buffer")).IsMemoryLike();
      }
      return obj.Has("ptr") and obj.Get("ptr").IsNumber();
    }
    return false;
  }

  inline bool IsDeviceMemoryLike() const {
    if (IsObject() and not IsNull()) {
      NapiToCPP::Object const& obj = val;
      if (obj.Has("buffer")) {  //
        return obj.Get("buffer").IsDeviceMemoryLike();
      }
      return obj.Has("ptr") and obj.Get("ptr").IsNumber();
    }
    return false;
  }

  inline Napi::Boolean ToBoolean() const { return val.ToBoolean(); }
  inline Napi::Number ToNumber() const { return val.ToNumber(); }
  inline Napi::String ToString() const { return val.ToString(); }
  inline Napi::Object ToObject() const { return val.ToObject(); }

  template <typename T>
  T As() const {
    return val.As<T>();
  }

  template <typename T>
  operator Wrapper<T>() const {
    return Wrapper<T>(As<Napi::Object>());
  }

  template <typename T>
  operator T() const;

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
  inline operator Napi::TypedArrayOf<T>() const {
    bool is_valid{false};
    std::size_t length{};
    std::size_t offset{};
    Napi::ArrayBuffer buffer;
    if (IsArrayBuffer()) {
      is_valid = true;
      buffer   = val.As<Napi::ArrayBuffer>();
      length   = buffer.ByteLength() / sizeof(T);
    } else if (IsDataView()) {
      is_valid       = true;
      auto const ary = val.As<Napi::DataView>();
      buffer         = ary.ArrayBuffer();
      offset         = ary.ByteOffset();
      length         = ary.ByteLength() / sizeof(T);
    } else if (IsTypedArray()) {
      is_valid       = true;
      auto const ary = val.As<Napi::TypedArray>();
      buffer         = ary.ArrayBuffer();
      offset         = ary.ByteOffset();
      length         = ary.ByteLength() / sizeof(T);
    }
    if (is_valid) { return Napi::TypedArrayOf<T>::New(Env(), length, buffer, offset); }
    throw Napi::Error::New(Env(), "Expected ArrayBuffer or ArrayBufferView");
  }

  template <typename T>
  inline operator Napi::Buffer<T>() const {
    return val.As<Napi::Buffer<T>>();
  }

  inline operator Object() const { return Object(val); }

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
      auto env = Env();
      NapiToCPP v{env.Null()};
      auto arr = val.As<Napi::Array>();
      std::vector<T> vec;
      vec.reserve(arr.Length());
      for (uint32_t i = 0; i < arr.Length(); ++i) {
        Napi::HandleScope scope{env};
        v.val = arr.Get(i);
        vec.push_back(v);
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
      auto env = Env();
      NapiToCPP k{env.Null()};
      NapiToCPP v{env.Null()};
      std::map<Key, Val> map{};
      auto obj  = val.As<Napi::Object>();
      auto keys = obj.GetPropertyNames();
      for (uint32_t i = 0; i < keys.Length(); ++i) {
        Napi::HandleScope scope{env};
        k.val  = keys.Get(i);
        v.val  = obj.Get(k.val);
        map[k] = v;
      }
      return map;
    }
    return std::map<Key, Val>{};
  }

  //
  // Pointers
  //
  inline operator void*() const {        //
    return static_cast<void*>(as_span<char>().data());
  }
  inline operator void const*() const {  //
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

#ifdef GLEW_VERSION
  inline operator GLsync() const { return reinterpret_cast<GLsync>(this->operator char*()); }
#endif

#ifdef GLFW_APIENTRY_DEFINED
  inline operator GLFWcursor*() const {
    return reinterpret_cast<GLFWcursor*>(this->operator char*());
  }
  inline operator GLFWwindow*() const {
    return reinterpret_cast<GLFWwindow*>(this->operator char*());
  }
  inline operator GLFWmonitor*() const {
    return reinterpret_cast<GLFWmonitor*>(this->operator char*());
  }
  inline operator GLFWglproc*() const {
    return reinterpret_cast<GLFWglproc*>(this->operator char*());
  }
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
      auto span = NapiToCPP(obj).as_span<T>() + offset;
      if (span.data() != nullptr) {  //
        return Span<T>(span.data(), std::min(span.size(), length));
      }
    }
    return Span<T>(static_cast<char*>(nullptr), 0);
  }

  template <typename T>
  inline T to_numeric() const {
    if (val.IsNull() || val.IsEmpty()) { return 0; }
    if (val.IsNumber() || val.IsString()) { return val.ToNumber(); }
    if (val.IsBigInt()) {
      bool lossless = true;
      return std::is_signed<T>() ? static_cast<T>(val.As<Napi::BigInt>().Int64Value(&lossless))
                                 : static_cast<T>(val.As<Napi::BigInt>().Uint64Value(&lossless));
    }
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
