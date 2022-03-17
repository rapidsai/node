// Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

namespace nv {

struct LasHeader {
  char file_signature[4];
  uint16_t file_source_id;
  uint16_t global_encoding;

  char version_major, version_minor;

  char system_identifier[32];
  char generating_software[32];

  uint16_t header_size;
  uint32_t point_data_offset, variable_length_records_count;

  char point_data_format_id;

  uint16_t point_data_size;
  uint32_t point_record_count;
  uint32_t points_by_return_count[5];

  double x_scale, y_scale, z_scale;
  double x_offset, y_offset, z_offset;
  double max_x, min_x;
  double max_y, min_y;
  double max_z, min_z;
};

const std::vector<std::string> PointDataFormatZeroColumnNames = {"x",
                                                                 "y",
                                                                 "z",
                                                                 "intensity",
                                                                 "bit_data",
                                                                 "classification",
                                                                 "scan_angle",
                                                                 "user_data",
                                                                 "point_source_id"};

struct PointDataFormatZero {
  int32_t x, y, z;
  uint16_t intensity;
  uint8_t bit_data, classification;
  char scan_angle;
  uint8_t user_data;
  uint16_t point_source_id;
};

const std::vector<std::string> PointDataFormatOneColumnNames = {"x",
                                                                "y",
                                                                "z",
                                                                "intensity",
                                                                "bit_data",
                                                                "classification",
                                                                "scan_angle",
                                                                "user_data",
                                                                "point_source_id",
                                                                "gps_time"};

struct PointDataFormatOne {
  int32_t x, y, z;
  uint16_t intensity;
  uint8_t bit_data, classification;
  char scan_angle;
  uint8_t user_data;
  uint16_t point_source_id;
  double gps_time;
};

const std::vector<std::string> PointDataFormatTwoColumnNames = {"x",
                                                                "y",
                                                                "z",
                                                                "intensity",
                                                                "bit_data",
                                                                "classification",
                                                                "scan_angle",
                                                                "user_data",
                                                                "point_source_id",
                                                                "red",
                                                                "green",
                                                                "blue"};

struct PointDataFormatTwo {
  int32_t x, y, z;
  uint16_t intensity;
  uint8_t bit_data, classification;
  char scan_angle;
  uint8_t user_data;
  uint16_t point_source_id;
  uint16_t red;
  uint16_t green;
  uint16_t blue;
};

const std::vector<std::string> PointDataFormatThreeColumnNames = {"x",
                                                                  "y",
                                                                  "z",
                                                                  "intensity",
                                                                  "bit_data",
                                                                  "classification",
                                                                  "scan_angle",
                                                                  "user_data",
                                                                  "point_source_id",
                                                                  "gps_time"};

struct PointDataFormatThree {
  int32_t x, y, z;
  uint16_t intensity;
  uint8_t bit_data, classification;
  char scan_angle;
  uint8_t user_data;
  uint16_t point_source_id;
  double gps_time;
  uint16_t red;
  uint16_t green;
  uint16_t blue;
};

std::tuple<std::vector<std::string>, std::unique_ptr<cudf::table>> read_las(
  const std::unique_ptr<cudf::io::datasource>& datasource,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource(),
  rmm::cuda_stream_view stream        = rmm::cuda_stream_default);

}  // namespace nv
