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

#include <node_rmm/device_buffer.hpp>

#include <cudf/types.hpp>

namespace nv {

/**
 * @brief Unwrap or create DeviceBuffer from an Napi::Value suitable as a Column's data buffer.
 *
 * * If `value` is already a DeviceBuffer, unwrap and return it.
 * * If `value` is an Array of JavaScript numbers, construct a DeviceBuffer of `double` values.
 * * If `value` is an Array of JavaScript bigints, construct a DeviceBuffer of `int64_t` values.
 * * If `value` is an ArrayBuffer, ArrayBufferView, CUDA Memory, or CUDA MemoryView, copy-construct
 *   and return a DeviceBuffer.
 *
 * @param env The active JavaScript environment.
 * @param value JavaScript value to unwrap or convert.
 * @param dtype The expected dtype of the input values.
 * @param mr Memory resource to use for the device memory allocation.
 * @param stream CUDA stream on which memory may be allocated if the memory
 * resource supports streams.
 */
DeviceBuffer::wrapper_t data_to_devicebuffer(
  Napi::Env const& env,
  Napi::Value const& value,
  cudf::data_type const& dtype,
  MemoryResource::wrapper_t const& mr,
  rmm::cuda_stream_view stream = rmm::cuda_stream_default);

inline DeviceBuffer::wrapper_t data_to_devicebuffer(
  Napi::Env const& env,
  Napi::Value const& value,
  cudf::data_type const& dtype,
  rmm::cuda_stream_view stream = rmm::cuda_stream_default) {
  return data_to_devicebuffer(env, value, dtype, MemoryResource::Current(env), stream);
}

/**
 * @brief Unwrap or create DeviceBuffer from an Napi::Value suitable as a Column's null mask.
 *
 * * If `value` is already a DeviceBuffer, unwrap and return it.
 * * If `value` is an Array, construct a DeviceBuffer of non-null values.
 * * If `value` is a Boolean, construct a DeviceBuffer of all valid or invalid bits.
 * * If `value` is an ArrayBuffer, ArrayBufferView, CUDA Memory, or CUDA MemoryView, copy-construct
 *   and return a DeviceBuffer.
 *
 * @param env The active JavaScript environment.
 * @param value JavaScript value to unwrap or convert.
 * @param size The number of elements to be represented by the mask.
 * @param mr Memory resource to use for the device memory allocation.
 * @param stream CUDA stream on which memory may be allocated if the memory
 * resource supports streams.
 */
DeviceBuffer::wrapper_t mask_to_null_bitmask(
  Napi::Env const& env,
  Napi::Value const& value,
  cudf::size_type const& size,
  MemoryResource::wrapper_t const& mr,
  rmm::cuda_stream_view stream = rmm::cuda_stream_default);

inline DeviceBuffer::wrapper_t mask_to_null_bitmask(
  Napi::Env const& env,
  Napi::Value const& value,
  cudf::size_type const& size,
  rmm::cuda_stream_view stream = rmm::cuda_stream_default) {
  return mask_to_null_bitmask(env, value, size, MemoryResource::Current(env), stream);
}

/**
 * @brief Create a DeviceBuffer from an Napi::Value of input data values suitable as a Column's null
 * mask. Sets null/undefined elements to false, everything else to true.
 *
 * * If `value` is already a DeviceBuffer, unwrap and return it.
 * * If `value` is an Array, construct a DeviceBuffer of non-null values.
 * * If `value` is a Boolean, construct a DeviceBuffer of all valid or invalid bits.
 * * If `value` is an ArrayBuffer, ArrayBufferView, CUDA Memory, or CUDA MemoryView, copy-construct
 *   and return a DeviceBuffer.
 *
 * @param env The active JavaScript environment.
 * @param value JavaScript value to unwrap or convert.
 * @param size The number of elements to be represented by the mask.
 * @param mr Memory resource to use for the device memory allocation.
 * @param stream CUDA stream on which memory may be allocated if the memory
 * resource supports streams.
 */
DeviceBuffer::wrapper_t data_to_null_bitmask(
  Napi::Env const& env,
  Napi::Value const& value,
  cudf::size_type const& size,
  MemoryResource::wrapper_t const& mr,
  rmm::cuda_stream_view stream = rmm::cuda_stream_default);

inline DeviceBuffer::wrapper_t data_to_null_bitmask(
  Napi::Env const& env,
  Napi::Value const& value,
  cudf::size_type const& size,
  rmm::cuda_stream_view stream = rmm::cuda_stream_default) {
  return data_to_null_bitmask(env, value, size, MemoryResource::Current(env), stream);
}

}  // namespace nv
