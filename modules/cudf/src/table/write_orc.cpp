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

#include <node_cudf/table.hpp>
#include <node_cudf/utilities/metadata.hpp>

#include <cudf/io/data_sink.hpp>
#include <cudf/io/orc.hpp>

namespace nv {

namespace {

cudf::io::orc_writer_options make_writer_options(Napi::Object const& options,
                                                 cudf::io::sink_info const& sink,
                                                 cudf::table_view const& table,
                                                 cudf::io::table_input_metadata* metadata) {
  auto has_opt = [&](std::string const& key) { return options.Has(key); };
  auto str_opt = [&](std::string const& key, std::string const& default_val) {
    return has_opt(key) ? options.Get(key).ToString().Utf8Value() : default_val;
  };
  auto bool_opt = [&](std::string const& key, bool default_val) {
    return has_opt(key) ? options.Get(key).ToBoolean() == true : default_val;
  };

  return std::move(cudf::io::orc_writer_options::builder(sink, table)
                     .metadata(metadata)
                     .enable_statistics(bool_opt("enableStatistics", true))
                     .compression(str_opt("compression", "none") == "snappy"
                                    ? cudf::io::compression_type::SNAPPY
                                    : cudf::io::compression_type::NONE)
                     .build());
}

}  // namespace

void Table::write_orc(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  CallbackArgs args{info};
  NODE_CUDF_EXPECT(args[1].IsObject(), "writeORC expects an Object of WriteORCOptions", env);

  std::string file_path = args[0];
  auto options          = args[1].As<Napi::Object>();

  cudf::table_view table = *this;
  auto metadata          = make_writer_columns_metadata(options, table);
  auto writer_opts = make_writer_options(options, cudf::io::sink_info{file_path}, table, &metadata);
  cudf::io::write_orc(writer_opts);
}

}  // namespace nv
