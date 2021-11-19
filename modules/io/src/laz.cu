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
#include <laz.hpp>

#include <cudf/io/datasource.hpp>

__global__ void parse_header(uint8_t const* laz_header_data, LazHeader* result) {
  size_t byte_offset = 0;

  // File signature (4 bytes)
  for (int i = 0; i < 4; ++i) { result->file_signature[i] = *(laz_header_data + i); }
  byte_offset += 4;

  // File source id (2 bytes)
  result->file_source_id = *(laz_header_data + byte_offset) | *(laz_header_data + byte_offset + 1)
                                                                << 8;
  byte_offset += 2;

  // Global encoding (2 bytes)
  result->global_encoding = *(laz_header_data + byte_offset) | *(laz_header_data + byte_offset + 1)
                                                                 << 8;
  byte_offset += 2;

  // Project ID (16 bytes)
  // not required
  byte_offset += 16;

  // Version major (1 byte)
  result->version_major = *(laz_header_data + byte_offset);
  byte_offset += 1;

  // Version minor (1 byte)
  result->version_minor = *(laz_header_data + byte_offset);
  byte_offset += 1;

  // System identifier (32 bytes)
  for (int i = 0; i < 32; ++i) {
    result->system_identifier[i] = *(laz_header_data + byte_offset + i);
  }
  byte_offset += 32;

  // Generating software (32 bytes)
  for (int i = 0; i < 32; ++i) {
    result->generating_software[i] = *(laz_header_data + byte_offset + i);
  }
  byte_offset += 32;

  // File creation day of year (2 bytes)
  // not required
  byte_offset += 2;

  // File creation year (2 bytes)
  // not required
  byte_offset += 2;

  // Header size (2 bytes)
  result->header_size = *(laz_header_data + byte_offset) | *(laz_header_data + byte_offset + 1)
                                                             << 8;
  byte_offset += 2;

  // Offset to point data (4 bytes)
  result->point_data_offset =
    *(laz_header_data + byte_offset) | *(laz_header_data + byte_offset + 1) << 8 |
    *(laz_header_data + byte_offset + 2) << 16 | *(laz_header_data + byte_offset + 3) << 24;
  byte_offset += 4;

  // Number of variable length records (4 bytes)
  result->variable_length_records_count =
    *(laz_header_data + byte_offset) | *(laz_header_data + byte_offset + 1) << 8 |
    *(laz_header_data + byte_offset + 2) << 16 | *(laz_header_data + byte_offset + 3) << 24;
  byte_offset += 4;

  // Point data format id (1 byte)
  result->point_data_format_id = *(laz_header_data + byte_offset);
  if (result->point_data_format_id & 128 || result->point_data_format_id & 64)
    result->point_data_format_id &= 127;
  byte_offset += 1;

  // Point data record length (2 bytes)
  result->point_data_size = *(laz_header_data + byte_offset) | *(laz_header_data + byte_offset + 1)
                                                                 << 8;
  byte_offset += 2;

  // Number of point records (4 bytes)
  result->point_record_count =
    *(laz_header_data + byte_offset) | *(laz_header_data + byte_offset + 1) << 8 |
    *(laz_header_data + byte_offset + 2) << 16 | *(laz_header_data + byte_offset + 3) << 24;
  byte_offset += 4;

  // Number of points by return (20 bytes)
  for (int i = 0; i < 4; ++i) {
    result->points_by_return_count[i] =
      *(laz_header_data + byte_offset) | *(laz_header_data + byte_offset + 1) << 8 |
      *(laz_header_data + byte_offset + 2) << 16 | *(laz_header_data + byte_offset + 3) << 24;
    byte_offset += 4;
  }

  // X scale factor (8 bytes)
  result->x_scale =
    *(laz_header_data + byte_offset) | *(laz_header_data + byte_offset + 1) << 8 |
    *(laz_header_data + byte_offset + 2) << 16 | *(laz_header_data + byte_offset + 3) << 24 |
    *(laz_header_data + byte_offset + 4) << 32 | *(laz_header_data + byte_offset + 5) << 40 |
    *(laz_header_data + byte_offset + 6) << 48 | *(laz_header_data + byte_offset + 7) << 56;
  byte_offset += 8;

  // Y scale factor (8 bytes)
  result->y_scale =
    *(laz_header_data + byte_offset) | *(laz_header_data + byte_offset + 1) << 8 |
    *(laz_header_data + byte_offset + 2) << 16 | *(laz_header_data + byte_offset + 3) << 24 |
    *(laz_header_data + byte_offset + 4) << 32 | *(laz_header_data + byte_offset + 5) << 40 |
    *(laz_header_data + byte_offset + 6) << 48 | *(laz_header_data + byte_offset + 7) << 56;
  byte_offset += 8;

  // Z scale factor (8 bytes)
  result->z_scale =
    *(laz_header_data + byte_offset) | *(laz_header_data + byte_offset + 1) << 8 |
    *(laz_header_data + byte_offset + 2) << 16 | *(laz_header_data + byte_offset + 3) << 24 |
    *(laz_header_data + byte_offset + 4) << 32 | *(laz_header_data + byte_offset + 5) << 40 |
    *(laz_header_data + byte_offset + 6) << 48 | *(laz_header_data + byte_offset + 7) << 56;
  byte_offset += 8;

  // X offset (8 bytes)
  result->x_offset =
    *(laz_header_data + byte_offset) | *(laz_header_data + byte_offset + 1) << 8 |
    *(laz_header_data + byte_offset + 2) << 16 | *(laz_header_data + byte_offset + 3) << 24 |
    *(laz_header_data + byte_offset + 4) << 32 | *(laz_header_data + byte_offset + 5) << 40 |
    *(laz_header_data + byte_offset + 6) << 48 | *(laz_header_data + byte_offset + 7) << 56;
  byte_offset += 8;

  // Y offset (8 bytes)
  result->y_offset =
    *(laz_header_data + byte_offset) | *(laz_header_data + byte_offset + 1) << 8 |
    *(laz_header_data + byte_offset + 2) << 16 | *(laz_header_data + byte_offset + 3) << 24 |
    *(laz_header_data + byte_offset + 4) << 32 | *(laz_header_data + byte_offset + 5) << 40 |
    *(laz_header_data + byte_offset + 6) << 48 | *(laz_header_data + byte_offset + 7) << 56;
  byte_offset += 8;

  // Z offset (8 bytes)
  result->z_offset =
    *(laz_header_data + byte_offset) | *(laz_header_data + byte_offset + 1) << 8 |
    *(laz_header_data + byte_offset + 2) << 16 | *(laz_header_data + byte_offset + 3) << 24 |
    *(laz_header_data + byte_offset + 4) << 32 | *(laz_header_data + byte_offset + 5) << 40 |
    *(laz_header_data + byte_offset + 6) << 48 | *(laz_header_data + byte_offset + 7) << 56;
  byte_offset += 8;

  // Max X (8 bytes)
  result->max_x =
    *(laz_header_data + byte_offset) | *(laz_header_data + byte_offset + 1) << 8 |
    *(laz_header_data + byte_offset + 2) << 16 | *(laz_header_data + byte_offset + 3) << 24 |
    *(laz_header_data + byte_offset + 4) << 32 | *(laz_header_data + byte_offset + 5) << 40 |
    *(laz_header_data + byte_offset + 6) << 48 | *(laz_header_data + byte_offset + 7) << 56;
  byte_offset += 8;

  // Min X (8 bytes)
  result->min_x =
    *(laz_header_data + byte_offset) | *(laz_header_data + byte_offset + 1) << 8 |
    *(laz_header_data + byte_offset + 2) << 16 | *(laz_header_data + byte_offset + 3) << 24 |
    *(laz_header_data + byte_offset + 4) << 32 | *(laz_header_data + byte_offset + 5) << 40 |
    *(laz_header_data + byte_offset + 6) << 48 | *(laz_header_data + byte_offset + 7) << 56;
  byte_offset += 8;

  // Max Y (8 bytes)
  result->max_y =
    *(laz_header_data + byte_offset) | *(laz_header_data + byte_offset + 1) << 8 |
    *(laz_header_data + byte_offset + 2) << 16 | *(laz_header_data + byte_offset + 3) << 24 |
    *(laz_header_data + byte_offset + 4) << 32 | *(laz_header_data + byte_offset + 5) << 40 |
    *(laz_header_data + byte_offset + 6) << 48 | *(laz_header_data + byte_offset + 7) << 56;
  byte_offset += 8;

  // Min Y (8 bytes)
  result->min_y =
    *(laz_header_data + byte_offset) | *(laz_header_data + byte_offset + 1) << 8 |
    *(laz_header_data + byte_offset + 2) << 16 | *(laz_header_data + byte_offset + 3) << 24 |
    *(laz_header_data + byte_offset + 4) << 32 | *(laz_header_data + byte_offset + 5) << 40 |
    *(laz_header_data + byte_offset + 6) << 48 | *(laz_header_data + byte_offset + 7) << 56;
  byte_offset += 8;

  // Max Z (8 bytes)
  result->max_z =
    *(laz_header_data + byte_offset) | *(laz_header_data + byte_offset + 1) << 8 |
    *(laz_header_data + byte_offset + 2) << 16 | *(laz_header_data + byte_offset + 3) << 24 |
    *(laz_header_data + byte_offset + 4) << 32 | *(laz_header_data + byte_offset + 5) << 40 |
    *(laz_header_data + byte_offset + 6) << 48 | *(laz_header_data + byte_offset + 7) << 56;
  byte_offset += 8;

  // Min Z (8 bytes)
  result->min_z =
    *(laz_header_data + byte_offset) | *(laz_header_data + byte_offset + 1) << 8 |
    *(laz_header_data + byte_offset + 2) << 16 | *(laz_header_data + byte_offset + 3) << 24 |
    *(laz_header_data + byte_offset + 4) << 32 | *(laz_header_data + byte_offset + 5) << 40 |
    *(laz_header_data + byte_offset + 6) << 48 | *(laz_header_data + byte_offset + 7) << 56;
}

__global__ void parse_variable_length_header(uint8_t const* laz_variable_header_data,
                                             LazVariableLengthHeader* result) {
  size_t byte_offset = 0;

  // Reserved (2 bytes)
  // not required
  byte_offset += 2;

  // User id (16 bytes)
  for (int i = 0; i < 16; ++i) {
    result->user_id[i] = *(laz_variable_header_data + byte_offset + i);
  }
  byte_offset += 16;

  // Record id (2 bytes)
  result->record_id =
    *(laz_variable_header_data + byte_offset) | *(laz_variable_header_data + byte_offset + 1) << 8;
  byte_offset += 2;

  // Record length after header (2 bytes)
  result->record_length_after_head =
    *(laz_variable_header_data + byte_offset) | *(laz_variable_header_data + byte_offset + 1) << 8;
  byte_offset += 2;

  // Description (32 bytes)
  // not required
  byte_offset += 32;
}

__global__ void parse_point_record(uint8_t const* point_data,
                                   LazHeader* header_data,
                                   PointRecord* result) {
  switch (header_data->point_data_format_id) {
    case 0: break;  // format 0
    case 1: break;  // format 1
    case 2: break;  // format 2
    case 3:         // format 3
      for (size_t i = 0; i < header_data->point_record_count; ++i) {
        size_t byte_offset = i * header_data->point_data_size;

        // x (4 bytes)
        result->x = *(point_data + byte_offset) | *(point_data + byte_offset + 1) << 8 |
                    *(point_data + byte_offset + 2) << 16 | *(point_data + byte_offset + 3) << 24;
        byte_offset += 4;

        // y (4 bytes)
        result->y = *(point_data + byte_offset) | *(point_data + byte_offset + 1) << 8 |
                    *(point_data + byte_offset + 2) << 16 | *(point_data + byte_offset + 3) << 24;
        byte_offset += 4;

        // z (4 bytes)
        result->z = *(point_data + byte_offset) | *(point_data + byte_offset + 1) << 8 |
                    *(point_data + byte_offset + 2) << 16 | *(point_data + byte_offset + 3) << 24;
        byte_offset += 4;

        // intensity (2 bytes)
        // not required
        byte_offset += 2;

        // return number (1 byte)
        // not required
        byte_offset += 1;

        // classification (1 byte)
        result->classification = *(point_data + byte_offset);
        byte_offset += 1;

        // Scan angle (1 byte)
        result->scan_angle = *(point_data + byte_offset);
        byte_offset += 1;

        // User data (1 byte)
        // not required
        byte_offset += 1;

        // Point source id (2 bytes)
        result->point_source_id = *(point_data + byte_offset) | *(point_data + byte_offset + 1)
                                                                  << 8;
        byte_offset += 2;

        // GPS time (8 bytes)
        result->gps_time =
          *(point_data + byte_offset) | *(point_data + byte_offset + 1) << 8 |
          *(point_data + byte_offset + 2) << 16 | *(point_data + byte_offset + 3) << 24 |
          *(point_data + byte_offset + 4) << 32 | *(point_data + byte_offset + 5) << 40 |
          *(point_data + byte_offset + 6) << 48 | *(point_data + byte_offset + 7) << 56;
        byte_offset += 8;

        // Red (2 bytes)
        result->red = *(point_data + byte_offset) | *(point_data + byte_offset + 1) << 8;
        byte_offset += 2;

        // Green (2 bytes)
        result->green = *(point_data + byte_offset) | *(point_data + byte_offset + 1) << 8;
        byte_offset += 2;

        // Blue (2 bytes)
        result->blue = *(point_data + byte_offset) | *(point_data + byte_offset + 1) << 8;
        byte_offset += 2;
      }
      break;
  }
}

void Laz::parse_header_host() {
  // Header parse
  const size_t header_size = 227;
  auto header_data         = read(0, header_size, rmm::cuda_stream_default);

  LazHeader *cpu_header, *gpu_header;
  cpu_header = (LazHeader*)malloc(sizeof(LazHeader));
  cudaMalloc((void**)&gpu_header, sizeof(LazHeader));

  ::parse_header<<<1, 1>>>(header_data->data(), gpu_header);

  cudaMemcpy(cpu_header, gpu_header, sizeof(LazHeader), cudaMemcpyDeviceToHost);

  // Variable header parse
  const size_t variable_header_size = 54;
  auto variable_header_data = read(header_size, variable_header_size, rmm::cuda_stream_default);

  LazVariableLengthHeader *cpu_variable_header, *gpu_variable_header;
  cpu_variable_header = (LazVariableLengthHeader*)malloc(sizeof(LazVariableLengthHeader));
  cudaMalloc((void**)&gpu_variable_header, sizeof(LazVariableLengthHeader));

  ::parse_variable_length_header<<<1, 1>>>(variable_header_data->data(), gpu_variable_header);

  // Point record parse
  auto point_data = read(cpu_header->point_data_offset,
                         cpu_header->point_record_count * cpu_header->point_data_size,
                         rmm::cuda_stream_default);

  PointRecord *cpu_point_record, *gpu_point_record;
  cpu_point_record = (PointRecord*)malloc(sizeof(PointRecord));
  cudaMalloc((void**)&gpu_point_record, sizeof(PointRecord));

  ::parse_point_record<<<1, 1>>>(point_data->data(), gpu_header, gpu_point_record);

  cudaMemcpy(cpu_point_record, gpu_point_record, sizeof(PointRecord), cudaMemcpyDeviceToHost);

  cudaMemcpy(cpu_variable_header,
             gpu_variable_header,
             sizeof(LazVariableLengthHeader),
             cudaMemcpyDeviceToHost);

  free(cpu_header);
  cudaFree(gpu_header);

  free(cpu_variable_header);
  cudaFree(gpu_variable_header);

  throw std::invalid_argument("end test");
}

std::unique_ptr<cudf::io::datasource::buffer> Laz::read(size_t offset,
                                                        size_t size,
                                                        rmm::cuda_stream_view stream) {
  if (_datasource->supports_device_read()) {
    return _datasource->device_read(offset, size, stream);
  }
  auto device_buffer = rmm::device_buffer(size, stream);
  CUDA_TRY(cudaMemcpyAsync(device_buffer.data(),
                           _datasource->host_read(offset, size)->data(),
                           size,
                           cudaMemcpyHostToDevice,
                           stream.value()));
  return cudf::io::datasource::buffer::create(std::move(device_buffer));
}
