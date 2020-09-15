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
#include <type_traits>

namespace nv {

template <typename T>
struct Span {
  Span(T* const& data, size_t const& size) : data_(data), size_(size) {}
  Span(void* const& data, size_t const& size) {
    this->data_ = reinterpret_cast<T*>(data);
    this->size_ = 1.0 / sizeof(T) * size;
  }

  template <typename R>
  Span(R* const& data, size_t const& size)
    : data_(reinterpret_cast<T*>(data)),  //
      size_(static_cast<float>(sizeof(R)) / sizeof(T) * size) {}

  Span(Napi::External<T> const& external) : Span(external.Data(), 0) {}
  Span(Napi::ArrayBuffer& buffer) : Span(buffer.Data(), buffer.ByteLength()) {}
  Span(Napi::ArrayBuffer const& buffer) : Span(*const_cast<Napi::ArrayBuffer*>(&buffer)) {}
  Span(Napi::DataView const& view) : Span(view.ArrayBuffer()) { *this += view.ByteOffset(); }
  Span(Napi::TypedArray const& array) : Span(array.ArrayBuffer()) { *this += array.ByteOffset(); }

  template <typename R>
  inline operator Span<R>() const {
    return Span<R>(data_, size_);
  }

  inline T* data() const { return data_; }
  inline operator T*() const { return data_; }

  inline size_t size() const { return size_; }
  inline operator size_t() const { return size_; }

  Span<T>& operator+=(size_t const& offset) {
    this->data_ += offset;
    this->size_ -= offset;
    return *this;
  }

  Span<T> inline operator+(size_t offset) {
    auto copy = *this;
    copy += offset;
    return copy;
  }

 private:
  T* data_{};
  size_t size_{};
};

}  // namespace nv
