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
#include <rmm/device_buffer.hpp>

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

struct PointRecord {
  long x;
  long y;
  long z;
  unsigned char classification;
  unsigned char scan_angle;
  unsigned short point_source_id;
  double gps_time;
  unsigned short red;
  unsigned short green;
  unsigned short blue;
};

class Las {
 public:
  Las(const std::string& path);

  std::unique_ptr<cudf::table> make_table_from_las(
    LasHeader* header,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource(),
    rmm::cuda_stream_view stream        = rmm::cuda_stream_default);

 private:
  std::unique_ptr<cudf::io::datasource> _datasource;

  std::unique_ptr<cudf::io::datasource::buffer> read(size_t offset,
                                                     size_t size,
                                                     rmm::cuda_stream_view stream);

  void parse_host();
  void parse_header_host(LasHeader* cpu_header, LasHeader* gpu_header);

  const size_t header_size          = 227;
  const size_t variable_header_size = 54;
};
