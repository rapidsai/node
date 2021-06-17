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

#include <node_cudf/utilities/buffer.hpp>
#include <node_cudf/utilities/error.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/bit.hpp>

namespace nv {

namespace {

bool is_device_memory(Napi::Object const& data) {
  return data.Has("ptr") and data.Get("ptr").IsNumber();
}

bool is_device_buffer_wrapper(Napi::Object const& data) {
  return data.Has("buffer") and data.Get("buffer").IsObject() and
         DeviceBuffer::IsInstance(data.Get("buffer"));
}

bool is_device_memory_wrapper(Napi::Object const& data) {
  return data.Has("buffer") and data.Get("buffer").IsObject() and
         is_device_memory(data.Get("buffer").ToObject());
}

std::size_t get_size_type(Napi::Object const& data, std::string const& key) {
  if (data.Has(key)) {
    auto val = data.Get(key);
    if (val.IsNumber()) { return val.ToNumber().Int64Value(); }
    if (val.IsBigInt()) {
      bool lossless{false};
      return val.As<Napi::BigInt>().Uint64Value(&lossless);
    }
  }
  return 0;
}

char* get_device_memory_ptr(Napi::Object const& buffer) {
  return reinterpret_cast<char*>(buffer.Get("ptr").ToNumber().Int64Value());
}

DeviceBuffer::wrapper_t device_memory_to_device_buffer(Napi::Env const& env,
                                                       Napi::Object const& data,
                                                       MemoryResource::wrapper_t const& mr,
                                                       rmm::cuda_stream_view stream) {
  Napi::HandleScope scope{env};
  auto dptr   = get_device_memory_ptr(data);
  auto length = get_size_type(data, "byteLength");
  return DeviceBuffer::New(env, Span{dptr, length}, mr, stream);
}

DeviceBuffer::wrapper_t device_memory_wrapper_to_device_buffer(Napi::Env const& env,
                                                               Napi::Object const& data,
                                                               MemoryResource::wrapper_t const& mr,
                                                               rmm::cuda_stream_view stream) {
  Napi::HandleScope scope{env};
  auto length = get_size_type(data, "byteLength");
  auto offset = get_size_type(data, "byteOffset");
  auto dptr   = get_device_memory_ptr(data.Get("buffer").ToObject());
  return DeviceBuffer::New(env, Span{dptr + offset, length}, mr, stream);
}

DeviceBuffer::wrapper_t data_view_to_device_buffer(Napi::Env const& env,
                                                   Napi::Object const& data,
                                                   MemoryResource::wrapper_t const& mr,
                                                   rmm::cuda_stream_view stream) {
  Napi::HandleScope scope{env};
  auto dv  = data.As<Napi::DataView>();
  auto ary = Napi::Uint8Array::New(env, dv.ByteLength(), dv.ArrayBuffer(), dv.ByteOffset());
  return DeviceBuffer::New(env, ary, mr, stream);
}

DeviceBuffer::wrapper_t array_buffer_to_device_buffer(Napi::Env const& env,
                                                      Napi::Object const& data,
                                                      MemoryResource::wrapper_t const& mr,
                                                      rmm::cuda_stream_view stream) {
  Napi::HandleScope scope{env};
  auto buf = data.As<Napi::ArrayBuffer>();
  auto ary = Napi::Uint8Array::New(env, buf.ByteLength(), buf, 0);
  return DeviceBuffer::New(env, ary, mr, stream);
}

DeviceBuffer::wrapper_t typed_array_to_device_buffer(Napi::Env const& env,
                                                     Napi::Object const& data,
                                                     MemoryResource::wrapper_t const& mr,
                                                     rmm::cuda_stream_view stream) {
  Napi::HandleScope scope{env};
  auto length = get_size_type(data, "byteLength");
  auto offset = get_size_type(data, "byteOffset");
  auto buf    = data.Get("buffer").As<Napi::ArrayBuffer>();
  auto ary    = Napi::Uint8Array::New(env, length, buf, offset);
  return DeviceBuffer::New(env, ary, mr, stream);
}

DeviceBuffer::wrapper_t array_to_device_buffer(Napi::Env const& env,
                                               Napi::Array const& data,
                                               cudf::data_type const& dtype,
                                               MemoryResource::wrapper_t const& mr,
                                               rmm::cuda_stream_view stream) {
  switch (dtype.id()) {
    case cudf::type_id::INT64: return DeviceBuffer::New<int64_t>(env, data, mr);
    case cudf::type_id::UINT64: return DeviceBuffer::New<uint64_t>(env, data, mr);
    case cudf::type_id::FLOAT64: return DeviceBuffer::New<double>(env, data, mr);
    case cudf::type_id::FLOAT32:
    case cudf::type_id::INT8:
    case cudf::type_id::INT16:
    case cudf::type_id::INT32:
    case cudf::type_id::UINT8:
    case cudf::type_id::UINT16:
    case cudf::type_id::UINT32:
    case cudf::type_id::BOOL8: {
      auto buffer          = DeviceBuffer::New<double>(env, data, mr);
      cudf::size_type size = buffer->size() / sizeof(double);
      cudf::column_view view{cudf::data_type{cudf::type_id::FLOAT64}, size, buffer->data()};
      return DeviceBuffer::New(env, std::move(cudf::cast(view, dtype)->release().data), mr);
    }
    default: return DeviceBuffer::New(env, mr, stream);
  }
}

DeviceBuffer::wrapper_t bool_array_to_null_bitmask(Napi::Env const& env,
                                                   Napi::Array const& data,
                                                   cudf::size_type const& size,
                                                   MemoryResource::wrapper_t const& mr,
                                                   rmm::cuda_stream_view stream) {
  auto const mask_size = cudf::bitmask_allocation_size_bytes(size);
  std::vector<cudf::bitmask_type> mask(mask_size / sizeof(cudf::bitmask_type), 0);
  for (auto i = 0u; i < data.Length(); ++i) {
    Napi::HandleScope scope{env};
    // Set the valid bit if the value is "truthy" by JS standards
    if (data.Get(i).ToBoolean().Value()) { cudf::set_bit_unsafe(mask.data(), i); }
  }
  return DeviceBuffer::New(env, mask.data(), mask_size, MemoryResource::Current(env));
}

DeviceBuffer::wrapper_t data_array_to_null_bitmask(Napi::Env const& env,
                                                   Napi::Array const& data,
                                                   cudf::size_type const& size,
                                                   MemoryResource::wrapper_t const& mr,
                                                   rmm::cuda_stream_view stream) {
  auto const mask_size = cudf::bitmask_allocation_size_bytes(size);
  std::vector<cudf::bitmask_type> mask(mask_size / sizeof(cudf::bitmask_type), 0);
  for (auto i = 0u; i < data.Length(); ++i) {
    Napi::HandleScope scope{env};
    auto const elt = data.Get(i);
    // Set the valid bit if the value isn't `null` or `undefined`
    if (!(elt.IsNull() or elt.IsUndefined() or elt.IsEmpty())) {
      cudf::set_bit_unsafe(mask.data(), i);
    }
  }
  return DeviceBuffer::New(env, mask.data(), mask_size, MemoryResource::Current(env));
}

}  // namespace

DeviceBuffer::wrapper_t data_to_devicebuffer(Napi::Env const& env,
                                             Napi::Value const& value,
                                             cudf::data_type const& dtype,
                                             MemoryResource::wrapper_t const& mr,
                                             rmm::cuda_stream_view stream) {
  if (value.IsObject() and !(value.IsEmpty() || value.IsNull() || value.IsUndefined())) {
    auto data = value.As<Napi::Object>();
    if (DeviceBuffer::IsInstance(data)) { return data; }
    if (is_device_buffer_wrapper(data)) { return data.Get("buffer").ToObject(); }
    if (is_device_memory(data)) { return device_memory_to_device_buffer(env, data, mr, stream); }
    if (is_device_memory_wrapper(data)) {
      return device_memory_wrapper_to_device_buffer(env, data, mr, stream);
    }
    if (data.IsArrayBuffer()) { return array_buffer_to_device_buffer(env, data, mr, stream); }
    if (data.IsDataView()) { return data_view_to_device_buffer(env, data, mr, stream); }
    if (data.IsBuffer() || data.IsTypedArray()) {
      return typed_array_to_device_buffer(env, data, mr, stream);
    }
    if (data.IsArray()) {
      return array_to_device_buffer(env, data.As<Napi::Array>(), dtype, mr, stream);
    }
  }
  return DeviceBuffer::New(env, mr, stream);
}

DeviceBuffer::wrapper_t mask_to_null_bitmask(Napi::Env const& env,
                                             Napi::Value const& value,
                                             cudf::size_type const& size,
                                             MemoryResource::wrapper_t const& mr,
                                             rmm::cuda_stream_view stream) {
  if (size <= 0 || value.IsEmpty() || value.IsNull() || value.IsUndefined()) {
    // Return an empty bitmask indicating all-valid/non-nullable
    return DeviceBuffer::New(env, mr, stream);
  }
  if (value.IsBoolean()) {
    // Return a full bitmask indicating either all-valid or all-null
    auto mask = cudf::create_null_mask(
      size, value.ToBoolean() ? cudf::mask_state::ALL_VALID : cudf::mask_state::ALL_NULL);
    return DeviceBuffer::New(value.Env(), std::make_unique<rmm::device_buffer>(std::move(mask)));
  }
  if (value.IsArray()) {
    return bool_array_to_null_bitmask(env, value.As<Napi::Array>(), size, mr, stream);
  }
  if (value.IsObject()) {
    auto mask = data_to_devicebuffer(
      env, value.As<Napi::Object>(), cudf::data_type{cudf::type_id::BOOL8}, mr, stream);
    if (mask->size() > 0 && mask->size() < cudf::bitmask_allocation_size_bytes(size)) {
      mask->resize(cudf::bitmask_allocation_size_bytes(size), mask->stream());
    }
    NODE_CUDF_EXPECT(mask->size() == 0 || mask->size() >= cudf::bitmask_allocation_size_bytes(size),
                     "Null mask buffer size must match the size of the column.",
                     env);
    return mask;
  }
  // Return an empty bitmask indicating all-valid/non-nullable
  return DeviceBuffer::New(env, mr, stream);
}

DeviceBuffer::wrapper_t data_to_null_bitmask(Napi::Env const& env,
                                             Napi::Value const& value,
                                             cudf::size_type const& size,
                                             MemoryResource::wrapper_t const& mr,
                                             rmm::cuda_stream_view stream) {
  if (size <= 0 || value.IsEmpty() || value.IsNull() || value.IsUndefined()) {
    // Return an empty bitmask indicating all-valid/non-nullable
    return DeviceBuffer::New(env, mr, stream);
  }
  if (value.IsArray()) {
    return data_array_to_null_bitmask(env, value.As<Napi::Array>(), size, mr, stream);
  }
  // Return an empty bitmask indicating all-valid/non-nullable
  return DeviceBuffer::New(env, mr, stream);
}

}  // namespace nv
