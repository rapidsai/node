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

#include <napi.h>

#include <iostream>
#include <map>
#include <string>
#include <tuple>
#include <type_traits>

#include <node_webgl/gl.hpp>

namespace node_webgl {

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

struct FromJS {
  Napi::Value val;
  inline FromJS(const Napi::Value& val) : val(val) {}

  inline std::ostream& operator<<(std::ostream& os) const {
    return os << this->operator std::string();
  }

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
  template <typename T>
  inline operator Napi::Reference<T>() const {
    return val.As<Napi::Reference<T>>();
  }
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

  template <typename T, typename R = T>
  T inline as_numeric() const {
    if (val.IsNull()) { return 0; }
    if (val.IsObject()) {
      if (val.IsArray()) {
        auto arr = val.As<Napi::Array>();
        if (arr.Length() == 1) {
          auto elt = arr.Get(static_cast<uint32_t>(0));
          if (elt.IsNumber()) { return static_cast<T>(static_cast<R>(elt.ToNumber())); }
        }
      } else {
        auto obj = val.As<Napi::Object>();
        if (obj.Has("_")) {
          auto ptr = obj.Get("_");
          if (ptr.IsNumber()) { return static_cast<T>(static_cast<R>(ptr.ToNumber())); }
        }
      }
    }
    return static_cast<T>(static_cast<R>(val.ToNumber()));
  }

  inline operator float() const { return this->as_numeric<float, float>(); }
  inline operator double() const { return this->as_numeric<double, double>(); }
  inline operator int8_t() const { return this->as_numeric<int8_t, int64_t>(); }
  inline operator int16_t() const { return this->as_numeric<int16_t, int64_t>(); }
  inline operator int32_t() const { return this->as_numeric<int32_t, int32_t>(); }
  inline operator int64_t() const { return this->as_numeric<int64_t, int64_t>(); }
  inline operator uint8_t() const { return this->as_numeric<uint8_t, int64_t>(); }
  inline operator uint16_t() const { return this->as_numeric<uint16_t, int64_t>(); }
  inline operator uint32_t() const { return this->as_numeric<uint32_t, uint32_t>(); }
  inline operator uint64_t() const { return this->as_numeric<uint64_t, int64_t>(); }
  inline operator std::string() const { return val.ToString().operator std::string(); }
  inline operator std::u16string() const { return val.ToString().operator std::u16string(); }

  // inline operator char*() const {
  //   std::string str = FromJS(val);
  //   auto ptr = reinterpret_cast<char*>(malloc(str.size()));
  //   memcpy(ptr, str.c_str(), str.size());
  //   return ptr;
  // }

  // inline operator CUresult() const {
  //   return static_cast<CUresult>(this->val.ToNumber().Int32Value());
  // }
  // inline operator CUstream() const {
  //   return reinterpret_cast<CUstream>(this->val.ToNumber().Int64Value());
  // }
  // inline operator CUdevice_attribute() const {
  //   return
  //   static_cast<CUdevice_attribute>(this->val.ToNumber().Uint32Value());
  // }
  // inline operator CUpointer_attribute() const {
  //   return
  //   static_cast<CUpointer_attribute>(this->val.ToNumber().Uint32Value());
  // }

  // inline operator GLFWcursor*() const {
  //   return reinterpret_cast<GLFWcursor*>(this->val.ToNumber().Uint32Value());
  // }
  // inline operator GLFWwindow*() const {
  //   return reinterpret_cast<GLFWwindow*>(this->val.ToNumber().Uint32Value());
  // }
  // inline operator GLFWmonitor*() const {
  //   return reinterpret_cast<GLFWmonitor*>(this->val.ToNumber().Uint32Value());
  // }
  // inline operator GLFWglproc*() const {
  //   return reinterpret_cast<GLFWglproc*>(this->val.ToNumber().Uint32Value());
  // }

  //
  // Arrays
  //
  // template <typename Lhs, typename Rhs>
  // inline operator std::pair<Lhs, Rhs>() const {
  //   if (val.IsArray()) {
  //     auto arr = val.As<Napi::Array>();
  //     auto lhs = arr.Get(static_cast<uint32_t>(0));
  //     auto rhs = arr.Get(static_cast<uint32_t>(1));
  //     return std::make_pair<Lhs, Rhs>(FromJS(lhs).operator Lhs(), FromJS(rhs).operator Rhs());
  //   }
  //   return std::make_pair<Lhs, Rhs>(Lhs{}, Rhs{});
  // }

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
    if (!(val.IsNull() || val.IsEmpty())) { return std::vector<T>{this->operator T()}; }
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

#define STRUCT_CONVERSION_OPERATOR(T)                                          \
  inline operator T() const {                                                  \
    T props{};                                                                 \
    if (val.IsObject()) {                                                      \
      auto obj = val.As<Napi::Object>();                                       \
      visit_struct::for_each(props, [&](const char* key, auto& val) {          \
        using R = typename std::decay<decltype(val)>::type;                    \
        if (!obj.Has(key) || obj.Get(key).IsUndefined()) { return; }           \
        if (!std::is_pointer<R>()) {                                           \
          R* dst = reinterpret_cast<R*>(&val);                                 \
          *dst   = FromJS(obj.Get(key)).operator R();                          \
        } else {                                                               \
          using R_ = typename std::remove_pointer<R>::type;                    \
          R_* src  = reinterpret_cast<R_*>(FromJS(obj.Get(key)).operator R()); \
          memcpy(reinterpret_cast<R_*>(val), src, sizeof(val));                \
          free(src);                                                           \
        }                                                                      \
      });                                                                      \
    }                                                                          \
    return props;                                                              \
  }

  // STRUCT_CONVERSION_OPERATOR(GLFWAPI::GLFWimage)
  // STRUCT_CONVERSION_OPERATOR(GLFWAPI::GLFWvidmode)
  // STRUCT_CONVERSION_OPERATOR(GLFWAPI::GLFWgammaramp)
  // STRUCT_CONVERSION_OPERATOR(GLFWAPI::GLFWgamepadstate)

#undef STRUCT_CONVERSION_OPERATOR

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
    if (val.IsNumber()) { return reinterpret_cast<void*>(val.ToNumber().operator int64_t()); }
    // if (val.IsObject()) {
    //   auto obj = val.As<Napi::Object>();
    //   if (obj.Has("buffer")) {
    //     obj = obj.Get("buffer").As<Napi::Object>();
    //   }
    //   if (obj.Has("byteLength")) {
    //     try {
    //       return mem::CUDABuffer::Unwrap(obj)->Data();
    //     } catch (Napi::Error e) {
    //     }
    //   }
    // }
    return reinterpret_cast<void*>(val.operator napi_value());
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

  inline operator GLsync() const { return reinterpret_cast<GLsync>(this->operator void*()); }
};

struct ToNapi {
  Napi::Env env;
  inline ToNapi(Napi::Env const& env) : env(env) {}

  Napi::Boolean inline operator()(const Napi::Boolean& val) const { return val; }
  Napi::Number inline operator()(const Napi::Number& val) const { return val; }
  Napi::String inline operator()(const Napi::String& val) const { return val; }
  Napi::Object inline operator()(const Napi::Object& val) const { return val; }
  Napi::Array inline operator()(const Napi::Array& val) const { return val; }
  Napi::Function inline operator()(const Napi::Function& val) const { return val; }
  Napi::Error inline operator()(const Napi::Error& val) const { return val; }
  Napi::TypedArray inline operator()(const Napi::TypedArray& val) const { return val; }
  Napi::ArrayBuffer inline operator()(const Napi::ArrayBuffer& val) const { return val; }
  Napi::Value inline operator()(const Napi::Value& val) const { return val; }

  // WebGLActiveInfo inline operator()(const WebGLActiveInfo& val) { return val; }
  // WebGLShaderPrecisionFormat inline operator()(const WebGLShaderPrecisionFormat& val) { return
  // val; } WebGLBuffer inline operator()(const WebGLBuffer& val) { return val; } WebGLContextEvent
  // inline operator()(const WebGLContextEvent& val) { return val; } WebGLFramebuffer inline
  // operator()(const WebGLFramebuffer& val) { return val; } WebGLProgram inline operator()(const
  // WebGLProgram& val) { return val; } WebGLQuery inline operator()(const WebGLQuery& val) { return
  // val; } WebGLRenderbuffer inline operator()(const WebGLRenderbuffer& val) { return val; }
  // WebGLSampler inline operator()(const WebGLSampler& val) { return val; }
  // WebGLShader inline operator()(const WebGLShader& val) { return val; }
  // WebGLSync inline operator()(const WebGLSync& val) { return val; }
  // WebGLTexture inline operator()(const WebGLTexture& val) { return val; }
  // WebGLTransformFeedback inline operator()(const WebGLTransformFeedback& val) { return val; }
  // WebGLUniformLocation inline operator()(const WebGLUniformLocation& val) { return val; }
  // WebGLVertexArrayObject inline operator()(const WebGLVertexArrayObject& val) { return val; }

  template <typename T>
  Napi::Reference<T> inline operator()(const Napi::Reference<T>& val) {
    return val;
  }
  napi_value inline operator()(const napi_value& val) const { return val; }

  // Primitives
  Napi::Boolean inline operator()(const bool val) const {
    return Napi::Boolean::New(this->env, val);
  }
  Napi::Number inline operator()(const float val) const {
    return Napi::Number::New(this->env, val);
  }
  Napi::Number inline operator()(const double val) const {
    return Napi::Number::New(this->env, val);
  }
  Napi::Number inline operator()(const int8_t val) const {
    return Napi::Number::New(this->env, val);
  }
  Napi::Number inline operator()(const int16_t val) const {
    return Napi::Number::New(this->env, val);
  }
  Napi::Number inline operator()(const int32_t val) const {
    return Napi::Number::New(this->env, val);
  }
  Napi::Number inline operator()(const int64_t val) const {
    return Napi::Number::New(this->env, val);
  }
  Napi::Number inline operator()(const uint8_t val) const {
    return Napi::Number::New(this->env, val);
  }
  Napi::Number inline operator()(const uint16_t val) const {
    return Napi::Number::New(this->env, val);
  }
  Napi::Number inline operator()(const uint32_t val) const {
    return Napi::Number::New(this->env, val);
  }
  Napi::Number inline operator()(const uint64_t val) const {
    return Napi::Number::New(this->env, val);
  }
  Napi::String inline operator()(const char* val) const {
    return Napi::String::New(this->env, (val == NULL) ? "" : val);
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

  template <typename... Vals>
  Napi::Object inline operator()(const std::initializer_list<std::string>& keys,
                                 const std::tuple<Vals...>& vals) const {
    auto cast_t = *this;
    auto key    = keys.begin();
    auto obj    = Napi::Object::New(this->env);
    node_webgl::for_each(vals, [&](auto val) {
      obj.Set(cast_t(*key), cast_t(val));
      std::advance(key, 1);
    });
    return obj;
  }

  template <typename T>
  Napi::Object inline operator()(const std::initializer_list<std::string>& keys,
                                 const std::initializer_list<T>& vals) const {
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

#define STRUCT_CONVERSION_OPERATOR(T)                                                     \
  Napi::Object inline operator()(const T& props) const {                                  \
    auto cast_t = *this;                                                                  \
    auto obj    = Napi::Object::New(this->env);                                           \
    visit_struct::for_each(                                                               \
      props, [&](const char* name, const auto& value) { obj.Set(name, cast_t(value)); }); \
    return obj;                                                                           \
  }

  // STRUCT_CONVERSION_OPERATOR(GLFWAPI::GLFWimage)
  // STRUCT_CONVERSION_OPERATOR(GLFWAPI::GLFWvidmode)
  // STRUCT_CONVERSION_OPERATOR(GLFWAPI::GLFWgammaramp)
  // STRUCT_CONVERSION_OPERATOR(GLFWAPI::GLFWgamepadstate)

#undef STRUCT_CONVERSION_OPERATOR

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
  Napi::Value inline operator()(void* ptr) const {
    return (ptr == NULL) ? this->env.Null()
                         : Napi::Number::New(this->env, reinterpret_cast<size_t>(ptr));
  }
  // Napi::Uint8Array inline operator()(const cudaUUID_t& val) const {
  //   auto arr = Napi::Uint8Array::New(this->env, sizeof(cudaUUID_t));
  //   memcpy(arr.ArrayBuffer().Data(), &val.bytes, sizeof(cudaUUID_t));
  //   return arr;
  // }
  // Napi::Uint8Array inline operator()(const CUipcMemHandle& ptr) const {
  //   auto arr = Napi::Uint8Array::New(this->env, CU_IPC_HANDLE_SIZE);
  //   memcpy(arr.ArrayBuffer().Data(), &ptr.reserved, CU_IPC_HANDLE_SIZE);
  //   return arr;
  // }
  // Napi::Uint8Array inline operator()(const cudaIpcMemHandle_t& ptr) const {
  //   auto arr = Napi::Uint8Array::New(this->env, CU_IPC_HANDLE_SIZE);
  //   memcpy(arr.ArrayBuffer().Data(), &ptr.reserved, CU_IPC_HANDLE_SIZE);
  //   return arr;
  // }
  Napi::Value inline operator()(GLsync val) const {
    return (val == NULL) ? this->env.Null() : Napi::External<void>::New(this->env, val);
  }
  // Napi::Value inline operator()(CUfunction val) const {
  //   return (val == NULL) ? this->env.Null()
  //                        : Napi::External<void>::New(this->env, val);
  // }
  // Napi::Value inline operator()(nvrtcProgram val) const {
  //   return (val == NULL) ? this->env.Null()
  //                        : Napi::External<void>::New(this->env, val);
  // }
  // Napi::Value inline operator()(CUgraphicsResource val) const {
  //   return (val == NULL) ? this->env.Null()
  //                        : Napi::External<void>::New(this->env, val);
  // }
  // Napi::Value inline operator()(GLFWcursor* val) const {
  //   return Napi::Number::New(this->env, static_cast<uint32_t>(reinterpret_cast<size_t>(val)));
  // }
  // Napi::Value inline operator()(GLFWwindow* val) const {
  //   return Napi::Number::New(this->env, static_cast<uint32_t>(reinterpret_cast<size_t>(val)));
  // }
  // Napi::Value inline operator()(GLFWmonitor* val) const {
  //   return Napi::Number::New(this->env, static_cast<uint32_t>(reinterpret_cast<size_t>(val)));
  // }
};

}  // namespace node_webgl
