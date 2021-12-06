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

#include <iostream>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/io/datasource.hpp>

#include <las.hpp>

__global__ void parse_header(uint8_t const* las_header_data, LasHeader* result) {
  size_t byte_offset = 0;

  // File signature (4 bytes)
  for (int i = 0; i < 4; ++i) { result->file_signature[i] = *(las_header_data + i); }
  byte_offset += 4;

  // File source id (2 bytes)
  result->file_source_id = *(las_header_data + byte_offset) | *(las_header_data + byte_offset + 1)
                                                                << 8;
  byte_offset += 2;

  // Global encoding (2 bytes)
  result->global_encoding = *(las_header_data + byte_offset) | *(las_header_data + byte_offset + 1)
                                                                 << 8;
  byte_offset += 2;

  // Project ID (16 bytes)
  // not required
  byte_offset += 16;

  // Version major (1 byte)
  result->version_major = *(las_header_data + byte_offset);
  byte_offset += 1;

  // Version minor (1 byte)
  result->version_minor = *(las_header_data + byte_offset);
  byte_offset += 1;

  // System identifier (32 bytes)
  for (int i = 0; i < 32; ++i) {
    result->system_identifier[i] = *(las_header_data + byte_offset + i);
  }
  byte_offset += 32;

  // Generating software (32 bytes)
  for (int i = 0; i < 32; ++i) {
    result->generating_software[i] = *(las_header_data + byte_offset + i);
  }
  byte_offset += 32;

  // File creation day of year (2 bytes)
  // not required
  byte_offset += 2;

  // File creation year (2 bytes)
  // not required
  byte_offset += 2;

  // Header size (2 bytes)
  result->header_size = *(las_header_data + byte_offset) | *(las_header_data + byte_offset + 1)
                                                             << 8;
  byte_offset += 2;

  // Offset to point data (4 bytes)
  result->point_data_offset =
    *(las_header_data + byte_offset) | *(las_header_data + byte_offset + 1) << 8 |
    *(las_header_data + byte_offset + 2) << 16 | *(las_header_data + byte_offset + 3) << 24;
  byte_offset += 4;

  // Number of variable length records (4 bytes)
  result->variable_length_records_count =
    *(las_header_data + byte_offset) | *(las_header_data + byte_offset + 1) << 8 |
    *(las_header_data + byte_offset + 2) << 16 | *(las_header_data + byte_offset + 3) << 24;
  byte_offset += 4;

  // Point data format id (1 byte)
  result->point_data_format_id = *(las_header_data + byte_offset);
  if (result->point_data_format_id & 128 || result->point_data_format_id & 64)
    result->point_data_format_id &= 127;
  byte_offset += 1;

  // Point data record length (2 bytes)
  result->point_data_size = *(las_header_data + byte_offset) | *(las_header_data + byte_offset + 1)
                                                                 << 8;
  byte_offset += 2;

  // Number of point records (4 bytes)
  result->point_record_count =
    *(las_header_data + byte_offset) | *(las_header_data + byte_offset + 1) << 8 |
    *(las_header_data + byte_offset + 2) << 16 | *(las_header_data + byte_offset + 3) << 24;
  byte_offset += 4;

  // Number of points by return (20 bytes)
  for (int i = 0; i < 4; ++i) {
    result->points_by_return_count[i] =
      *(las_header_data + byte_offset) | *(las_header_data + byte_offset + 1) << 8 |
      *(las_header_data + byte_offset + 2) << 16 | *(las_header_data + byte_offset + 3) << 24;
    byte_offset += 4;
  }

  // X scale factor (8 bytes)
  result->x_scale =
    *(las_header_data + byte_offset) | *(las_header_data + byte_offset + 1) << 8 |
    *(las_header_data + byte_offset + 2) << 16 | *(las_header_data + byte_offset + 3) << 24 |
    *(las_header_data + byte_offset + 4) << 32 | *(las_header_data + byte_offset + 5) << 40 |
    *(las_header_data + byte_offset + 6) << 48 | *(las_header_data + byte_offset + 7) << 56;
  byte_offset += 8;

  // Y scale factor (8 bytes)
  result->y_scale =
    *(las_header_data + byte_offset) | *(las_header_data + byte_offset + 1) << 8 |
    *(las_header_data + byte_offset + 2) << 16 | *(las_header_data + byte_offset + 3) << 24 |
    *(las_header_data + byte_offset + 4) << 32 | *(las_header_data + byte_offset + 5) << 40 |
    *(las_header_data + byte_offset + 6) << 48 | *(las_header_data + byte_offset + 7) << 56;
  byte_offset += 8;

  // Z scale factor (8 bytes)
  result->z_scale =
    *(las_header_data + byte_offset) | *(las_header_data + byte_offset + 1) << 8 |
    *(las_header_data + byte_offset + 2) << 16 | *(las_header_data + byte_offset + 3) << 24 |
    *(las_header_data + byte_offset + 4) << 32 | *(las_header_data + byte_offset + 5) << 40 |
    *(las_header_data + byte_offset + 6) << 48 | *(las_header_data + byte_offset + 7) << 56;
  byte_offset += 8;

  // X offset (8 bytes)
  result->x_offset =
    *(las_header_data + byte_offset) | *(las_header_data + byte_offset + 1) << 8 |
    *(las_header_data + byte_offset + 2) << 16 | *(las_header_data + byte_offset + 3) << 24 |
    *(las_header_data + byte_offset + 4) << 32 | *(las_header_data + byte_offset + 5) << 40 |
    *(las_header_data + byte_offset + 6) << 48 | *(las_header_data + byte_offset + 7) << 56;
  byte_offset += 8;

  // Y offset (8 bytes)
  result->y_offset =
    *(las_header_data + byte_offset) | *(las_header_data + byte_offset + 1) << 8 |
    *(las_header_data + byte_offset + 2) << 16 | *(las_header_data + byte_offset + 3) << 24 |
    *(las_header_data + byte_offset + 4) << 32 | *(las_header_data + byte_offset + 5) << 40 |
    *(las_header_data + byte_offset + 6) << 48 | *(las_header_data + byte_offset + 7) << 56;
  byte_offset += 8;

  // Z offset (8 bytes)
  result->z_offset =
    *(las_header_data + byte_offset) | *(las_header_data + byte_offset + 1) << 8 |
    *(las_header_data + byte_offset + 2) << 16 | *(las_header_data + byte_offset + 3) << 24 |
    *(las_header_data + byte_offset + 4) << 32 | *(las_header_data + byte_offset + 5) << 40 |
    *(las_header_data + byte_offset + 6) << 48 | *(las_header_data + byte_offset + 7) << 56;
  byte_offset += 8;

  // Max X (8 bytes)
  result->max_x =
    *(las_header_data + byte_offset) | *(las_header_data + byte_offset + 1) << 8 |
    *(las_header_data + byte_offset + 2) << 16 | *(las_header_data + byte_offset + 3) << 24 |
    *(las_header_data + byte_offset + 4) << 32 | *(las_header_data + byte_offset + 5) << 40 |
    *(las_header_data + byte_offset + 6) << 48 | *(las_header_data + byte_offset + 7) << 56;
  byte_offset += 8;

  // Min X (8 bytes)
  result->min_x =
    *(las_header_data + byte_offset) | *(las_header_data + byte_offset + 1) << 8 |
    *(las_header_data + byte_offset + 2) << 16 | *(las_header_data + byte_offset + 3) << 24 |
    *(las_header_data + byte_offset + 4) << 32 | *(las_header_data + byte_offset + 5) << 40 |
    *(las_header_data + byte_offset + 6) << 48 | *(las_header_data + byte_offset + 7) << 56;
  byte_offset += 8;

  // Max Y (8 bytes)
  result->max_y =
    *(las_header_data + byte_offset) | *(las_header_data + byte_offset + 1) << 8 |
    *(las_header_data + byte_offset + 2) << 16 | *(las_header_data + byte_offset + 3) << 24 |
    *(las_header_data + byte_offset + 4) << 32 | *(las_header_data + byte_offset + 5) << 40 |
    *(las_header_data + byte_offset + 6) << 48 | *(las_header_data + byte_offset + 7) << 56;
  byte_offset += 8;

  // Min Y (8 bytes)
  result->min_y =
    *(las_header_data + byte_offset) | *(las_header_data + byte_offset + 1) << 8 |
    *(las_header_data + byte_offset + 2) << 16 | *(las_header_data + byte_offset + 3) << 24 |
    *(las_header_data + byte_offset + 4) << 32 | *(las_header_data + byte_offset + 5) << 40 |
    *(las_header_data + byte_offset + 6) << 48 | *(las_header_data + byte_offset + 7) << 56;
  byte_offset += 8;

  // Max Z (8 bytes)
  result->max_z =
    *(las_header_data + byte_offset) | *(las_header_data + byte_offset + 1) << 8 |
    *(las_header_data + byte_offset + 2) << 16 | *(las_header_data + byte_offset + 3) << 24 |
    *(las_header_data + byte_offset + 4) << 32 | *(las_header_data + byte_offset + 5) << 40 |
    *(las_header_data + byte_offset + 6) << 48 | *(las_header_data + byte_offset + 7) << 56;
  byte_offset += 8;

  // Min Z (8 bytes)
  result->min_z =
    *(las_header_data + byte_offset) | *(las_header_data + byte_offset + 1) << 8 |
    *(las_header_data + byte_offset + 2) << 16 | *(las_header_data + byte_offset + 3) << 24 |
    *(las_header_data + byte_offset + 4) << 32 | *(las_header_data + byte_offset + 5) << 40 |
    *(las_header_data + byte_offset + 6) << 48 | *(las_header_data + byte_offset + 7) << 56;
}

std::unique_ptr<cudf::io::datasource::buffer> read(
  const std::unique_ptr<cudf::io::datasource>& datasource,
  size_t offset,
  size_t size,
  rmm::cuda_stream_view stream) {
  if (datasource->supports_device_read()) { return datasource->device_read(offset, size, stream); }
  auto device_buffer = rmm::device_buffer(size, stream);
  CUDA_TRY(cudaMemcpyAsync(device_buffer.data(),
                           datasource->host_read(offset, size)->data(),
                           size,
                           cudaMemcpyHostToDevice,
                           stream.value()));
  return cudf::io::datasource::buffer::create(std::move(device_buffer));
}

std::unique_ptr<cudf::table> get_point_cloud_records(
  const std::unique_ptr<cudf::io::datasource>& datasource,
  LasHeader* cpu_header,
  rmm::mr::device_memory_resource* mr,
  rmm::cuda_stream_view stream) {
  auto const& point_record_count = cpu_header->point_record_count;
  auto const& point_data_offset  = cpu_header->point_data_offset;
  auto const& point_data_size    = cpu_header->point_data_size;

  auto point_data =
    read(datasource, point_data_offset, point_data_size * point_record_count, stream);

  auto data = point_data->data();
  auto idxs = thrust::make_counting_iterator(0);
  std::vector<std::unique_ptr<cudf::column>> cols;

  switch (cpu_header->point_data_format_id) {
    // POINT
    // FORMAT
    // ZERO
    case 0: {
      cols.resize(9);

      std::vector<cudf::type_id> ids{{
        cudf::type_id::INT32,  // x
        cudf::type_id::INT32,  // y
        cudf::type_id::INT32,  // z
        cudf::type_id::INT16,  // intensity
        cudf::type_id::INT8,   // bit_data
        cudf::type_id::INT8,   // classification
        cudf::type_id::INT8,   // scan angle
        cudf::type_id::INT8,   // user data
        cudf::type_id::INT16,  // point source id
      }};

      std::transform(ids.begin(), ids.end(), cols.begin(), [&](auto const& type_id) {
        return cudf::make_numeric_column(
          cudf::data_type{type_id}, point_record_count, cudf::mask_state::UNALLOCATED, stream, mr);
      });

      auto iter = thrust::make_transform_iterator(idxs, [=] __host__ __device__(int const& i) {
        auto ptr = data + (i * (point_data_size));
        PointDataFormatZero point_data;
        point_data.x               = static_cast<int32_t const>(*(ptr + 0));
        point_data.y               = static_cast<int32_t const>(*(ptr + 4));
        point_data.z               = static_cast<int32_t const>(*(ptr + 8));
        point_data.intensity       = static_cast<int16_t const>(*(ptr + 12));
        point_data.bit_data        = static_cast<int8_t const>(*(ptr + 14));
        point_data.classification  = static_cast<int8_t const>(*(ptr + 15));
        point_data.scan_angle      = static_cast<int8_t const>(*(ptr + 16));
        point_data.user_data       = static_cast<int8_t const>(*(ptr + 17));
        point_data.point_source_id = static_cast<int16_t const>(*(ptr + 18));
        return thrust::make_tuple(point_data.x,
                                  point_data.y,
                                  point_data.z,
                                  point_data.intensity,
                                  point_data.bit_data,
                                  point_data.classification,
                                  point_data.scan_angle,
                                  point_data.user_data,
                                  point_data.point_source_id);
      });

      thrust::copy(
        rmm::exec_policy(stream),
        iter,
        iter + point_record_count,
        thrust::make_zip_iterator(cols[0]->mutable_view().begin<int32_t>(),    // x
                                  cols[1]->mutable_view().begin<int32_t>(),    // y
                                  cols[2]->mutable_view().begin<int32_t>(),    // z
                                  cols[3]->mutable_view().begin<int16_t>(),    // intensity
                                  cols[4]->mutable_view().begin<int8_t>(),     // bits
                                  cols[5]->mutable_view().begin<int8_t>(),     // classification
                                  cols[6]->mutable_view().begin<int8_t>(),     // scan angle
                                  cols[7]->mutable_view().begin<int8_t>(),     // user data
                                  cols[8]->mutable_view().begin<int16_t>()));  // point source id
      break;
    }

    // POINT
    // FORMAT
    // ONE
    case 1: {
      cols.resize(10);

      std::vector<cudf::type_id> ids{{
        cudf::type_id::INT32,    // x
        cudf::type_id::INT32,    // y
        cudf::type_id::INT32,    // z
        cudf::type_id::INT16,    // intensity
        cudf::type_id::INT8,     // bit_data
        cudf::type_id::INT8,     // classification
        cudf::type_id::INT8,     // scan angle
        cudf::type_id::INT8,     // user data
        cudf::type_id::INT16,    // point source id
        cudf::type_id::FLOAT64,  // gps time
      }};

      std::transform(ids.begin(), ids.end(), cols.begin(), [&](auto const& type_id) {
        return cudf::make_numeric_column(
          cudf::data_type{type_id}, point_record_count, cudf::mask_state::UNALLOCATED, stream, mr);
      });

      auto iter = thrust::make_transform_iterator(idxs, [=] __host__ __device__(int const& i) {
        auto ptr = data + (i * (point_data_size));
        PointDataFormatOne point_data;
        point_data.x               = static_cast<int32_t const>(*(ptr + 0));
        point_data.y               = static_cast<int32_t const>(*(ptr + 4));
        point_data.z               = static_cast<int32_t const>(*(ptr + 8));
        point_data.intensity       = static_cast<int16_t const>(*(ptr + 12));
        point_data.bit_data        = static_cast<int8_t const>(*(ptr + 14));
        point_data.classification  = static_cast<int8_t const>(*(ptr + 15));
        point_data.scan_angle      = static_cast<int8_t const>(*(ptr + 16));
        point_data.user_data       = static_cast<int8_t const>(*(ptr + 17));
        point_data.point_source_id = static_cast<int16_t const>(*(ptr + 18));
        point_data.gps_time        = static_cast<double_t const>(*(ptr + 20));
        return thrust::make_tuple(point_data.x,
                                  point_data.y,
                                  point_data.z,
                                  point_data.intensity,
                                  point_data.bit_data,
                                  point_data.classification,
                                  point_data.scan_angle,
                                  point_data.user_data,
                                  point_data.point_source_id,
                                  point_data.gps_time);
      });

      thrust::copy(
        rmm::exec_policy(stream),
        iter,
        iter + point_record_count,
        thrust::make_zip_iterator(cols[0]->mutable_view().begin<int32_t>(),     // x
                                  cols[1]->mutable_view().begin<int32_t>(),     // y
                                  cols[2]->mutable_view().begin<int32_t>(),     // z
                                  cols[3]->mutable_view().begin<int16_t>(),     // intensity
                                  cols[4]->mutable_view().begin<int8_t>(),      // bits
                                  cols[5]->mutable_view().begin<int8_t>(),      // classification
                                  cols[6]->mutable_view().begin<int8_t>(),      // scan angle
                                  cols[7]->mutable_view().begin<int8_t>(),      // user data
                                  cols[8]->mutable_view().begin<int16_t>(),     // point source id
                                  cols[9]->mutable_view().begin<double_t>()));  // gps time
      break;
    }

    // POINT
    // FORMAT
    // THREE
    // TODO: Missing colours
    case 2: {
      cols.resize(9);

      std::vector<cudf::type_id> ids{{
        cudf::type_id::INT32,  // x
        cudf::type_id::INT32,  // y
        cudf::type_id::INT32,  // z
        cudf::type_id::INT16,  // intensity
        cudf::type_id::INT8,   // bit_data
        cudf::type_id::INT8,   // classification
        cudf::type_id::INT8,   // scan angle
        cudf::type_id::INT8,   // user data
        cudf::type_id::INT16,  // point source id
      }};

      std::transform(ids.begin(), ids.end(), cols.begin(), [&](auto const& type_id) {
        return cudf::make_numeric_column(
          cudf::data_type{type_id}, point_record_count, cudf::mask_state::UNALLOCATED, stream, mr);
      });

      auto iter = thrust::make_transform_iterator(idxs, [=] __host__ __device__(int const& i) {
        auto ptr = data + (i * (point_data_size));
        PointDataFormatTwo point_data;
        point_data.x               = static_cast<int32_t const>(*(ptr + 0));
        point_data.y               = static_cast<int32_t const>(*(ptr + 4));
        point_data.z               = static_cast<int32_t const>(*(ptr + 8));
        point_data.intensity       = static_cast<int16_t const>(*(ptr + 12));
        point_data.bit_data        = static_cast<int8_t const>(*(ptr + 14));
        point_data.classification  = static_cast<int8_t const>(*(ptr + 15));
        point_data.scan_angle      = static_cast<int8_t const>(*(ptr + 16));
        point_data.user_data       = static_cast<int8_t const>(*(ptr + 17));
        point_data.point_source_id = static_cast<int16_t const>(*(ptr + 18));
        return thrust::make_tuple(point_data.x,
                                  point_data.y,
                                  point_data.z,
                                  point_data.intensity,
                                  point_data.bit_data,
                                  point_data.classification,
                                  point_data.scan_angle,
                                  point_data.user_data,
                                  point_data.point_source_id);
      });

      thrust::copy(
        rmm::exec_policy(stream),
        iter,
        iter + point_record_count,
        thrust::make_zip_iterator(cols[0]->mutable_view().begin<int32_t>(),    // x
                                  cols[1]->mutable_view().begin<int32_t>(),    // y
                                  cols[2]->mutable_view().begin<int32_t>(),    // z
                                  cols[3]->mutable_view().begin<int16_t>(),    // intensity
                                  cols[4]->mutable_view().begin<int8_t>(),     // bits
                                  cols[5]->mutable_view().begin<int8_t>(),     // classification
                                  cols[6]->mutable_view().begin<int8_t>(),     // scan angle
                                  cols[7]->mutable_view().begin<int8_t>(),     // user data
                                  cols[8]->mutable_view().begin<int16_t>()));  // point source id
      break;
    }

    // POINT
    // FORMAT
    // THREE
    // TODO: Missing colours
    case 3: {
      cols.resize(10);

      std::vector<cudf::type_id> ids{{
        cudf::type_id::INT32,    // x
        cudf::type_id::INT32,    // y
        cudf::type_id::INT32,    // z
        cudf::type_id::INT16,    // intensity
        cudf::type_id::INT8,     // bit_data
        cudf::type_id::INT8,     // classification
        cudf::type_id::INT8,     // scan angle
        cudf::type_id::INT8,     // user data
        cudf::type_id::INT16,    // point source id
        cudf::type_id::FLOAT64,  // gps time
      }};

      std::transform(ids.begin(), ids.end(), cols.begin(), [&](auto const& type_id) {
        return cudf::make_numeric_column(
          cudf::data_type{type_id}, point_record_count, cudf::mask_state::UNALLOCATED, stream, mr);
      });

      auto iter = thrust::make_transform_iterator(idxs, [=] __host__ __device__(int const& i) {
        auto ptr = data + (i * (point_data_size));
        PointDataFormatThree point_data;
        point_data.x               = static_cast<int32_t const>(*(ptr + 0));
        point_data.y               = static_cast<int32_t const>(*(ptr + 4));
        point_data.z               = static_cast<int32_t const>(*(ptr + 8));
        point_data.intensity       = static_cast<int16_t const>(*(ptr + 12));
        point_data.bit_data        = static_cast<int8_t const>(*(ptr + 14));
        point_data.classification  = static_cast<int8_t const>(*(ptr + 15));
        point_data.scan_angle      = static_cast<int8_t const>(*(ptr + 16));
        point_data.user_data       = static_cast<int8_t const>(*(ptr + 17));
        point_data.point_source_id = static_cast<int16_t const>(*(ptr + 18));
        point_data.gps_time        = static_cast<double_t const>(*(ptr + 20));
        return thrust::make_tuple(point_data.x,
                                  point_data.y,
                                  point_data.z,
                                  point_data.intensity,
                                  point_data.bit_data,
                                  point_data.classification,
                                  point_data.scan_angle,
                                  point_data.user_data,
                                  point_data.point_source_id,
                                  point_data.gps_time);
      });

      thrust::copy(
        rmm::exec_policy(stream),
        iter,
        iter + point_record_count,
        thrust::make_zip_iterator(cols[0]->mutable_view().begin<int32_t>(),     // x
                                  cols[1]->mutable_view().begin<int32_t>(),     // y
                                  cols[2]->mutable_view().begin<int32_t>(),     // z
                                  cols[3]->mutable_view().begin<int16_t>(),     // intensity
                                  cols[4]->mutable_view().begin<int8_t>(),      // bits
                                  cols[5]->mutable_view().begin<int8_t>(),      // classification
                                  cols[6]->mutable_view().begin<int8_t>(),      // scan angle
                                  cols[7]->mutable_view().begin<int8_t>(),      // user data
                                  cols[8]->mutable_view().begin<int16_t>(),     // point source id
                                  cols[9]->mutable_view().begin<double_t>()));  // gps time
      break;
    }
  }

  return std::make_unique<cudf::table>(std::move(cols));
}

void parse_las_header_host(const std::unique_ptr<cudf::io::datasource>& datasource,
                           LasHeader* cpu_header,
                           LasHeader* gpu_header,
                           rmm::cuda_stream_view stream) {
  auto header_data = read(datasource, 0, HEADER_BYTE_SIZE, stream);
  ::parse_header<<<1, 1>>>(header_data->data(), gpu_header);

  cudaMemcpy(cpu_header, gpu_header, sizeof(LasHeader), cudaMemcpyDeviceToHost);
}

std::unique_ptr<cudf::table> parse_las_host(const std::unique_ptr<cudf::io::datasource>& datasource,
                                            rmm::mr::device_memory_resource* mr,
                                            rmm::cuda_stream_view stream) {
  LasHeader *cpu_header, *gpu_header;
  cpu_header = (LasHeader*)malloc(sizeof(LasHeader));
  cudaMalloc((void**)&gpu_header, sizeof(LasHeader));
  parse_las_header_host(datasource, cpu_header, gpu_header, stream);

  auto table = get_point_cloud_records(datasource, cpu_header, mr, stream);

  free(cpu_header);
  cudaFree(gpu_header);

  return table;
}
