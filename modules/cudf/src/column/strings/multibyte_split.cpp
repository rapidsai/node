// Copyright (c) 2022, NVIDIA CORPORATION.
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

namespace nv {

namespace {

Column::wrapper_t split_string_column(Napi::CallbackInfo const& info,
                                      cudf::mutable_column_view const& col,
                                      std::string const& delimiter,
                                      rmm::mr::device_memory_resource* mr) {
  /* TODO: This only splits a string column. How to generalize */
  // Check type
  auto span = cudf::device_span<char const>(col.child(1).data<char const>(), col.child(1).size());
  auto datasource = cudf::io::text::device_span_data_chunk_source(span);
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
