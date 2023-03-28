// Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include <node_cudf/column.hpp>
#include <node_cudf/table.hpp>
#include <node_cudf/utilities/metadata.hpp>

#include <cudf/io/text/data_chunk_source_factories.hpp>
#include <cudf/io/text/multibyte_split.hpp>
#include <cudf/utilities/span.hpp>

namespace nv {

namespace {

class device_span_data_chunk : public cudf::io::text::device_data_chunk {
 public:
  device_span_data_chunk(cudf::device_span<char const> data) : _data(data) {}

  [[nodiscard]] char const* data() const override { return _data.data(); }
  [[nodiscard]] std::size_t size() const override { return _data.size(); }
  operator cudf::device_span<char const>() const override { return _data; }

 private:
  cudf::device_span<char const> _data;
};

/**
 * @brief A reader which produces view of device memory which represent a subset of the input device
 * span.
 */
class device_span_data_chunk_reader : public cudf::io::text::data_chunk_reader {
 public:
  device_span_data_chunk_reader(cudf::device_span<char const> data) : _data(data) {}

  void skip_bytes(std::size_t read_size) override {
    _position += std::min(read_size, _data.size() - _position);
  }

  std::unique_ptr<cudf::io::text::device_data_chunk> get_next_chunk(
    std::size_t read_size, rmm::cuda_stream_view stream) override {
    // limit the read size to the number of bytes remaining in the device_span.
    read_size = std::min(read_size, _data.size() - _position);

    // create a view over the device span
    auto chunk_span = _data.subspan(_position, read_size);

    // increment position
    _position += read_size;

    // return the view over device memory so it can be processed.
    return std::make_unique<device_span_data_chunk>(chunk_span);
  }

 private:
  cudf::device_span<char const> _data;
  uint64_t _position = 0;
};

/**
 * @brief A device span data source which creates an istream_data_chunk_reader.
 */
class device_span_data_chunk_source : public cudf::io::text::data_chunk_source {
 public:
  device_span_data_chunk_source(cudf::device_span<char const> data) : _data(data) {}
  [[nodiscard]] std::unique_ptr<cudf::io::text::data_chunk_reader> create_reader() const override {
    return std::make_unique<device_span_data_chunk_reader>(_data);
  }

 private:
  cudf::device_span<char const> _data;
};

Column::wrapper_t split_string_column(Napi::CallbackInfo const& info,
                                      cudf::mutable_column_view const& col,
                                      std::string const& delimiter,
                                      rmm::mr::device_memory_resource* mr) {
  /* TODO: This only splits a string column. How to generalize */
  // Check type
  auto span = cudf::device_span<char const>(col.child(1).data<char const>(), col.child(1).size());
  auto datasource = device_span_data_chunk_source(span);
  return Column::New(info.Env(),
                     cudf::io::text::multibyte_split(datasource, delimiter, std::nullopt, mr));
}

Column::wrapper_t read_text_files(Napi::CallbackInfo const& info,
                                  std::string const& filename,
                                  std::string const& delimiter,
                                  rmm::mr::device_memory_resource* mr) {
  auto datasource = cudf::io::text::make_source_from_file(filename);
  return Column::New(info.Env(),
                     cudf::io::text::multibyte_split(*datasource, delimiter, std::nullopt, mr));
}

}  // namespace

Napi::Value Column::split(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};
  std::string const delimiter         = args[0];
  rmm::mr::device_memory_resource* mr = args[1];
  try {
    return split_string_column(info, *this, delimiter, mr);
  } catch (std::exception const& e) { throw Napi::Error::New(info.Env(), e.what()); }
}

Napi::Value Column::read_text(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};
  std::string const source            = args[0];
  std::string const delimiter         = args[1];
  rmm::mr::device_memory_resource* mr = args[2];

  try {
    return read_text_files(info, source, delimiter, mr);
  } catch (std::exception const& e) { throw Napi::Error::New(info.Env(), e.what()); }
}

}  // namespace nv
