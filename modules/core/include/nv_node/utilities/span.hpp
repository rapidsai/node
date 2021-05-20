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

namespace nv {

template <typename T>
struct Span {
  Span(std::size_t size) : data_{nullptr}, size_{size} {}
  Span(T* data, std::size_t size) : data_{data}, size_{size} {}
  Span(void* data, std::size_t size) : Span(reinterpret_cast<T*>(data), size) {}

  template <typename R>
  Span(R* data, std::size_t size)
    : data_{reinterpret_cast<T*>(data)},  //
      size_{static_cast<size_t>(static_cast<double>(sizeof(R)) / sizeof(T) * size)} {}

  Span(Napi::External<T> const& external) : Span<T>(external.Data(), 0) {}
  Span(Napi::ArrayBuffer& buffer) : Span<T>(buffer.Data(), buffer.ByteLength() / sizeof(T)) {}
  Span(Napi::ArrayBuffer const& buffer) : Span<T>(*const_cast<Napi::ArrayBuffer*>(&buffer)) {}
  Span(Napi::DataView const& view) {
    this->data_ = reinterpret_cast<T*>(view.ArrayBuffer().Data()) + (view.ByteOffset() / sizeof(T));
    this->size_ = view.ByteLength() / sizeof(T);
  }
  Span(Napi::TypedArray const& ary) {
    this->data_ = reinterpret_cast<T*>(ary.ArrayBuffer().Data()) + (ary.ByteOffset() / sizeof(T));
    this->size_ = ary.ByteLength() / sizeof(T);
  }

  template <typename R>
  inline operator Span<R>() const {
    return Span<R>(data_, size_);
  }

  inline operator void*() const noexcept { return static_cast<void*>(data_); }

  inline T* data() const noexcept { return data_; }
  inline operator T*() const noexcept { return data_; }

  inline std::size_t size() const noexcept { return size_; }
  inline operator std::size_t() const noexcept { return size_; }

  inline std::size_t addr() const noexcept { return reinterpret_cast<std::size_t>(data_); }

  Span<T>& operator+=(std::size_t const offset) noexcept {
    if (data_ != nullptr && size_ > 0) {
      this->data_ += offset;
      this->size_ -= offset;
    }
    return *this;
  }

  Span<T> inline operator+(std::size_t offset) const noexcept {
    auto copy = *this;
    copy += offset;
    return copy;
  }

 private:
  T* data_{};
  std::size_t size_{};
};

}  // namespace nv
