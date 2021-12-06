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

#include <cudf/io/datasource.hpp>
#include <cudf/table/table.hpp>
#include <rmm/device_buffer.hpp>

const int HEADER_BYTE_SIZE = 227;

struct LasHeader {
  char file_signature[4];
  unsigned short file_source_id;
  unsigned short global_encoding;
  unsigned char version_major;
  unsigned char version_minor;
  char system_identifier[32];
  char generating_software[32];
  unsigned short header_size;
  unsigned long point_data_offset;
  unsigned long variable_length_records_count;
  unsigned char point_data_format_id;
  unsigned short point_data_size;
  unsigned long point_record_count;
  unsigned long points_by_return_count[5];
  double x_scale;
  double y_scale;
  double z_scale;
  double x_offset;
  double y_offset;
  double z_offset;
  double max_x;
  double min_x;
  double max_y;
  double min_y;
  double max_z;
  double min_z;
};

struct PointDataFormatZero {
  long x;
  long y;
  long z;
  unsigned short intensity;
  unsigned char bit_data;
  unsigned char classification;
  char scan_angle;
  unsigned char user_data;
  unsigned short point_source_id;
};

struct PointDataFormatOne {
  long x;
  long y;
  long z;
  unsigned short intensity;
  unsigned char bit_data;
  unsigned char classification;
  char scan_angle;
  unsigned char user_data;
  unsigned short point_source_id;
  double gps_time;
};

struct PointDataFormatTwo {
  long x;
  long y;
  long z;
  unsigned short intensity;
  unsigned char bit_data;
  unsigned char classification;
  char scan_angle;
  unsigned char user_data;
  unsigned short point_source_id;
  unsigned short red;
  unsigned short green;
  unsigned short blue;
};

struct PointDataFormatThree {
  long x;
  long y;
  long z;
  unsigned short intensity;
  unsigned char bit_data;
  unsigned char classification;
  char scan_angle;
  unsigned char user_data;
  unsigned short point_source_id;
  double gps_time;
  unsigned short red;
  unsigned short green;
  unsigned short blue;
};

std::unique_ptr<cudf::io::datasource::buffer> read(
  const std::unique_ptr<cudf::io::datasource>& datasource,
  size_t offset,
  size_t size,
  rmm::cuda_stream_view stream);

std::unique_ptr<cudf::table> get_point_cloud_records(
  const std::unique_ptr<cudf::io::datasource>& datasource,
  LasHeader* cpu_header,
  rmm::mr::device_memory_resource* mr,
  rmm::cuda_stream_view stream);

void parse_header_host(const std::unique_ptr<cudf::io::datasource>& datasource,
                       LasHeader* cpu_header,
                       LasHeader* gpu_header,
                       rmm::cuda_stream_view stream);

std::unique_ptr<cudf::table> parse_host(
  const std::unique_ptr<cudf::io::datasource>& datasource,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource(),
  rmm::cuda_stream_view stream        = rmm::cuda_stream_default);
