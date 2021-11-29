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
#include <las.hpp>

#include <cudf/io/datasource.hpp>

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

__global__ void parse_variable_length_header(uint8_t const* las_variable_header_data,
                                             LasHeader* gpu_header,
                                             LasVariableLengthHeader* result) {
  for (size_t i = 0; i < gpu_header->variable_length_records_count; ++i) {
    size_t byte_offset = i * 54;  // variable_header_size

    // Reserved (2 bytes)
    // not required
    byte_offset += 2;

    // User id (16 bytes)
    for (int i = 0; i < 16; ++i) {
      result[i].user_id[i] = *(las_variable_header_data + byte_offset + i);
    }
    byte_offset += 16;

    // Record id (2 bytes)
    result[i].record_id = *(las_variable_header_data + byte_offset) |
                          *(las_variable_header_data + byte_offset + 1) << 8;
    byte_offset += 2;

    // Record length after header (2 bytes)
    result[i].record_length_after_head = *(las_variable_header_data + byte_offset) |
                                         *(las_variable_header_data + byte_offset + 1) << 8;
    byte_offset += 2;

    // Description (32 bytes)
    // not required
    byte_offset += 32;
  }
}

__device__ void parse_point_record_format_0(uint8_t const* point_data,
                                            LasHeader* header_data,
                                            PointRecord* result) {
  size_t i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < header_data->point_record_count) {
    size_t byte_offset = i * header_data->point_data_size;

    // x (4 bytes)
    result[i].x = *(point_data + byte_offset) | *(point_data + byte_offset + 1) << 8 |
                  *(point_data + byte_offset + 2) << 16 | *(point_data + byte_offset + 3) << 24;
    byte_offset += 4;

    // y (4 bytes)
    result[i].y = *(point_data + byte_offset) | *(point_data + byte_offset + 1) << 8 |
                  *(point_data + byte_offset + 2) << 16 | *(point_data + byte_offset + 3) << 24;
    byte_offset += 4;

    // z (4 bytes)
    result[i].z = *(point_data + byte_offset) | *(point_data + byte_offset + 1) << 8 |
                  *(point_data + byte_offset + 2) << 16 | *(point_data + byte_offset + 3) << 24;
    byte_offset += 4;

    // intensity (2 bytes)
    // not required
    byte_offset += 2;

    // return number (1 byte)
    // not required
    byte_offset += 1;

    // classification (1 byte)
    result[i].classification = *(point_data + byte_offset);
    byte_offset += 1;

    // Scan angle (1 byte)
    result[i].scan_angle = *(point_data + byte_offset);
    byte_offset += 1;

    // User data (1 byte)
    // not required
    byte_offset += 1;

    // Point source id (2 bytes)
    result[i].point_source_id = *(point_data + byte_offset) | *(point_data + byte_offset + 1) << 8;
    byte_offset += 2;
  }
}

__device__ void parse_point_record_format_1(uint8_t const* point_data,
                                            LasHeader* header_data,
                                            PointRecord* result) {
  size_t i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < header_data->point_record_count) {
    size_t byte_offset = i * header_data->point_data_size;

    // x (4 bytes)
    result[i].x = *(point_data + byte_offset) | *(point_data + byte_offset + 1) << 8 |
                  *(point_data + byte_offset + 2) << 16 | *(point_data + byte_offset + 3) << 24;
    byte_offset += 4;

    // y (4 bytes)
    result[i].y = *(point_data + byte_offset) | *(point_data + byte_offset + 1) << 8 |
                  *(point_data + byte_offset + 2) << 16 | *(point_data + byte_offset + 3) << 24;
    byte_offset += 4;

    // z (4 bytes)
    result[i].z = *(point_data + byte_offset) | *(point_data + byte_offset + 1) << 8 |
                  *(point_data + byte_offset + 2) << 16 | *(point_data + byte_offset + 3) << 24;
    byte_offset += 4;

    // intensity (2 bytes)
    // not required
    byte_offset += 2;

    // return number (1 byte)
    // not required
    byte_offset += 1;

    // classification (1 byte)
    result[i].classification = *(point_data + byte_offset);
    byte_offset += 1;

    // Scan angle (1 byte)
    result[i].scan_angle = *(point_data + byte_offset);
    byte_offset += 1;

    // User data (1 byte)
    // not required
    byte_offset += 1;

    // Point source id (2 bytes)
    result[i].point_source_id = *(point_data + byte_offset) | *(point_data + byte_offset + 1) << 8;
    byte_offset += 2;

    // GPS time (8 bytes)
    result[i].gps_time =
      *(point_data + byte_offset) | *(point_data + byte_offset + 1) << 8 |
      *(point_data + byte_offset + 2) << 16 | *(point_data + byte_offset + 3) << 24 |
      *(point_data + byte_offset + 4) << 32 | *(point_data + byte_offset + 5) << 40 |
      *(point_data + byte_offset + 6) << 48 | *(point_data + byte_offset + 7) << 56;
    byte_offset += 8;
  }
}

__device__ void parse_point_record_format_2(uint8_t const* point_data,
                                            LasHeader* header_data,
                                            PointRecord* result) {
  size_t i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < header_data->point_record_count) {
    size_t byte_offset = i * header_data->point_data_size;

    // x (4 bytes)
    result[i].x = *(point_data + byte_offset) | *(point_data + byte_offset + 1) << 8 |
                  *(point_data + byte_offset + 2) << 16 | *(point_data + byte_offset + 3) << 24;
    byte_offset += 4;

    // y (4 bytes)
    result[i].y = *(point_data + byte_offset) | *(point_data + byte_offset + 1) << 8 |
                  *(point_data + byte_offset + 2) << 16 | *(point_data + byte_offset + 3) << 24;
    byte_offset += 4;

    // z (4 bytes)
    result[i].z = *(point_data + byte_offset) | *(point_data + byte_offset + 1) << 8 |
                  *(point_data + byte_offset + 2) << 16 | *(point_data + byte_offset + 3) << 24;
    byte_offset += 4;

    // intensity (2 bytes)
    // not required
    byte_offset += 2;

    // return number (1 byte)
    // not required
    byte_offset += 1;

    // classification (1 byte)
    result[i].classification = *(point_data + byte_offset);
    byte_offset += 1;

    // Scan angle (1 byte)
    result[i].scan_angle = *(point_data + byte_offset);
    byte_offset += 1;

    // User data (1 byte)
    // not required
    byte_offset += 1;

    // Point source id (2 bytes)
    result[i].point_source_id = *(point_data + byte_offset) | *(point_data + byte_offset + 1) << 8;
    byte_offset += 2;

    // Red (2 bytes)
    result[i].red = *(point_data + byte_offset) | *(point_data + byte_offset + 1) << 8;
    byte_offset += 2;

    // Green (2 bytes)
    result[i].green = *(point_data + byte_offset) | *(point_data + byte_offset + 1) << 8;
    byte_offset += 2;

    // Blue (2 bytes)
    result[i].blue = *(point_data + byte_offset) | *(point_data + byte_offset + 1) << 8;
    byte_offset += 2;
  }
}

__device__ void parse_point_record_format_3(uint8_t const* point_data,
                                            LasHeader* header_data,
                                            PointRecord* result) {
  size_t i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < header_data->point_record_count) {
    size_t byte_offset = i * header_data->point_data_size;

    // x (4 bytes)
    result[i].x = *(point_data + byte_offset) | *(point_data + byte_offset + 1) << 8 |
                  *(point_data + byte_offset + 2) << 16 | *(point_data + byte_offset + 3) << 24;
    byte_offset += 4;

    // y (4 bytes)
    result[i].y = *(point_data + byte_offset) | *(point_data + byte_offset + 1) << 8 |
                  *(point_data + byte_offset + 2) << 16 | *(point_data + byte_offset + 3) << 24;
    byte_offset += 4;

    // z (4 bytes)
    result[i].z = *(point_data + byte_offset) | *(point_data + byte_offset + 1) << 8 |
                  *(point_data + byte_offset + 2) << 16 | *(point_data + byte_offset + 3) << 24;
    byte_offset += 4;

    // intensity (2 bytes)
    // not required
    byte_offset += 2;

    // return number (1 byte)
    // not required
    byte_offset += 1;

    // classification (1 byte)
    result[i].classification = *(point_data + byte_offset);
    byte_offset += 1;

    // Scan angle (1 byte)
    result[i].scan_angle = *(point_data + byte_offset);
    byte_offset += 1;

    // User data (1 byte)
    // not required
    byte_offset += 1;

    // Point source id (2 bytes)
    result[i].point_source_id = *(point_data + byte_offset) | *(point_data + byte_offset + 1) << 8;
    byte_offset += 2;

    // GPS time (8 bytes)
    result[i].gps_time =
      *(point_data + byte_offset) | *(point_data + byte_offset + 1) << 8 |
      *(point_data + byte_offset + 2) << 16 | *(point_data + byte_offset + 3) << 24 |
      *(point_data + byte_offset + 4) << 32 | *(point_data + byte_offset + 5) << 40 |
      *(point_data + byte_offset + 6) << 48 | *(point_data + byte_offset + 7) << 56;
    byte_offset += 8;

    // Red (2 bytes)
    result[i].red = *(point_data + byte_offset) | *(point_data + byte_offset + 1) << 8;
    byte_offset += 2;

    // Green (2 bytes)
    result[i].green = *(point_data + byte_offset) | *(point_data + byte_offset + 1) << 8;
    byte_offset += 2;

    // Blue (2 bytes)
    result[i].blue = *(point_data + byte_offset) | *(point_data + byte_offset + 1) << 8;
    byte_offset += 2;
  }
}

__global__ void parse_point_record(uint8_t const* point_data,
                                   LasHeader* header_data,
                                   PointRecord* result) {
  switch (header_data->point_data_format_id) {
    case 0: parse_point_record_format_0(point_data, header_data, result); break;  // format 0
    case 1: parse_point_record_format_1(point_data, header_data, result); break;  // format 1
    case 2: parse_point_record_format_2(point_data, header_data, result); break;  // format 2
    case 3: parse_point_record_format_3(point_data, header_data, result); break;  // format 3
  }
}

void Las::parse_host() {
  LasHeader *cpu_header, *gpu_header;
  cpu_header = (LasHeader*)malloc(sizeof(LasHeader));
  cudaMalloc((void**)&gpu_header, sizeof(LasHeader));
  parse_header_host(cpu_header, gpu_header);

  LasVariableLengthHeader *cpu_variable_header, *gpu_variable_header;
  cpu_variable_header = (LasVariableLengthHeader*)malloc(cpu_header->variable_length_records_count *
                                                         sizeof(LasVariableLengthHeader));
  cudaMalloc((void**)&gpu_variable_header,
             cpu_header->variable_length_records_count * sizeof(LasVariableLengthHeader));
  parse_variable_header_host(cpu_header, gpu_header, cpu_variable_header, gpu_variable_header);

  PointRecord *cpu_point_record, *gpu_point_record;
  cpu_point_record = (PointRecord*)malloc(cpu_header->point_record_count * sizeof(PointRecord));
  cudaMalloc((void**)&gpu_point_record, cpu_header->point_record_count * sizeof(PointRecord));
  parse_point_records_host(cpu_header, gpu_header, cpu_point_record, gpu_point_record);

  free(cpu_header);
  cudaFree(gpu_header);

  free(cpu_variable_header);
  cudaFree(gpu_variable_header);

  free(cpu_point_record);
  cudaFree(gpu_point_record);

  throw std::invalid_argument("end test");
}

void Las::parse_header_host(LasHeader* cpu_header, LasHeader* gpu_header) {
  auto header_data = read(0, header_size, rmm::cuda_stream_default);
  ::parse_header<<<1, 1>>>(header_data->data(), gpu_header);

  cudaMemcpy(cpu_header, gpu_header, sizeof(LasHeader), cudaMemcpyDeviceToHost);
}

void Las::parse_variable_header_host(LasHeader* cpu_header,
                                     LasHeader* gpu_header,
                                     LasVariableLengthHeader* cpu_variable_header,
                                     LasVariableLengthHeader* gpu_variable_header) {
  // Bail out if we have nothing to parse.
  if (cpu_header->variable_length_records_count == 0) { return; }

  auto variable_header_data = read(header_size,
                                   cpu_header->variable_length_records_count * variable_header_size,
                                   rmm::cuda_stream_default);

  ::parse_variable_length_header<<<1, 1>>>(
    variable_header_data->data(), gpu_header, gpu_variable_header);

  cudaMemcpy(cpu_variable_header,
             gpu_variable_header,
             cpu_header->variable_length_records_count * sizeof(LasVariableLengthHeader),
             cudaMemcpyDeviceToHost);
}

void Las::parse_point_records_host(LasHeader* cpu_header,
                                   LasHeader* gpu_header,
                                   PointRecord* cpu_point_record,
                                   PointRecord* gpu_point_record) {
  auto point_data = read(cpu_header->point_data_offset,
                         cpu_header->point_data_size * cpu_header->point_record_count,
                         rmm::cuda_stream_default);

  int blockSize = 0, minGridSize = 0, gridSize = 0;
  cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, parse_point_record, 0, 0);
  gridSize = (cpu_header->point_record_count + blockSize - 1) / blockSize;

  ::parse_point_record<<<gridSize, blockSize>>>(point_data->data(), gpu_header, gpu_point_record);

  cudaMemcpy(cpu_point_record,
             gpu_point_record,
             cpu_header->point_record_count * sizeof(PointRecord),
             cudaMemcpyDeviceToHost);
}

std::unique_ptr<cudf::io::datasource::buffer> Las::read(size_t offset,
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
