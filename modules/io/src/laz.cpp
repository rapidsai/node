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

#include <laz.hpp>

Laz::Laz(const std::string& path) {
  _datasource  = ::cudf::io::datasource::create(path);
  _byte_offset = 0;

  parse_header_device();
}

std::unique_ptr<cudf::io::datasource::buffer> Laz::read(size_t offset,
                                                        size_t size,
                                                        rmm::cuda_stream_view stream) {
  this->_byte_offset += size;
  if (_datasource->supports_device_read()) {
    return _datasource->device_read(offset, size, stream);
  }
  auto host_read     = _datasource->host_read(offset, size)->data();
  auto device_buffer = rmm::device_buffer(size, stream);
  std::memcpy(device_buffer.data(), host_read, size);
  return cudf::io::datasource::buffer::create(std::move(device_buffer));
}

void Laz::parse_header() {
  // auto file_signature          = this->read_bytes(4);
  // this->_header.file_signature = *file_signature->data();

  // auto file_source_id          = this->read_bytes(2);
  // this->_header.file_source_id = *file_source_id->data();

  // auto global_encoding          = this->read_bytes(2);
  // this->_header.global_encoding = *global_encoding->data();

  // // Skip Project ID
  // this->read_bytes(16);

  // auto version_major          = this->read_bytes(1);
  // this->_header.version_major = *version_major->data();

  // auto version_minor          = this->read_bytes(1);
  // this->_header.version_minor = *version_minor->data();

  // auto system_identifier          = this->read_bytes(32);
  // this->_header.system_identifier = *system_identifier->data();

  // auto generating_software          = this->read_bytes(32);
  // this->_header.generating_software = *generating_software->data();

  // // Skip file creation
  // this->read_bytes(4);

  // auto header_size          = this->read_bytes(2);
  // this->_header.header_size = *header_size->data();

  // auto point_data_offset          = this->read_bytes(4);
  // this->_header.point_data_offset = *point_data_offset->data();

  // auto variable_length_records_count          = this->read_bytes(4);
  // this->_header.variable_length_records_count = *variable_length_records_count->data();

  // auto point_data_format_id          = this->read_bytes(1);
  // this->_header.point_data_format_id = *point_data_format_id->data();

  // auto point_data_record_length          = this->read_bytes(4);
  // this->_header.point_data_record_length = *point_data_record_length->data();

  // auto point_record_count          = this->read_bytes(4);
  // this->_header.point_record_count = *point_record_count->data();

  // // TODO
  // // unsigned long[5]
  // // number of points by return
  // this->read_bytes(20);

  // auto x_scale          = this->read_bytes(8);
  // this->_header.x_scale = *x_scale->data();

  // auto y_scale          = this->read_bytes(8);
  // this->_header.y_scale = *y_scale->data();

  // auto z_scale          = this->read_bytes(8);
  // this->_header.z_scale = *z_scale->data();

  // auto x_offset          = this->read_bytes(8);
  // this->_header.x_offset = *x_offset->data();

  // auto y_offset          = this->read_bytes(8);
  // this->_header.y_offset = *y_offset->data();

  // auto z_offset          = this->read_bytes(8);
  // this->_header.z_offset = *z_offset->data();

  // auto max_x          = this->read_bytes(8);
  // this->_header.max_x = *max_x->data();

  // auto min_x          = this->read_bytes(8);
  // this->_header.min_x = *min_x->data();

  // auto max_y          = this->read_bytes(8);
  // this->_header.max_y = *max_y->data();

  // auto min_y          = this->read_bytes(8);
  // this->_header.min_y = *min_y->data();

  // auto max_z          = this->read_bytes(8);
  // this->_header.max_z = *max_z->data();

  // auto min_z          = this->read_bytes(8);
  // this->_header.min_z = *min_z->data();
}

void Laz::parse_variable_header() {
  // // Skip reserved
  // this->read_bytes(2);

  // auto user_id                  = this->read_bytes(16);
  // this->_variableHeader.user_id = *user_id->data();

  // auto record_id                  = this->read_bytes(2);
  // this->_variableHeader.record_id = *record_id->data();

  // auto record_length_after_head                  = this->read_bytes(2);
  // this->_variableHeader.record_length_after_head = *record_length_after_head->data();

  // // Skip description
  // this->read_bytes(32);
}
