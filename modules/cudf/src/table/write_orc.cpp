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

#include <cudf/io/data_sink.hpp>
#include <cudf/io/orc.hpp>

namespace nv {

namespace {

cudf::io::orc_writer_options make_writer_options(Napi::Object const& options,
                                                 cudf::io::sink_info const& sink,
                                                 cudf::table_view const& table) {
  auto env      = options.Env();
  auto has_opt  = [&](std::string const& key) { return options.Has(key); };
  auto napi_opt = [&](std::string const& key) -> Napi::Value {
    return has_opt(key) ? options.Get(key) : env.Undefined();
  };
  auto str_opt = [&](std::string const& key, std::string const& default_val) {
    return has_opt(key) ? options.Get(key).ToString().Utf8Value() : default_val;
  };

  auto null_value = str_opt("nullValue", "N/A");
  cudf::io::table_input_metadata metadata{};
  Napi::Array column_names = napi_opt("columnNames").IsArray()
                               ? napi_opt("columnNames").As<Napi::Array>()
                               : Napi::Array::New(env, table.num_columns());
  metadata.column_metadata.reserve(table.num_columns());
  for (uint32_t i = 0; i < column_names.Length(); ++i) {
    auto name   = column_names.Has(i) ? column_names.Get(i) : env.Null();
    auto column = cudf::io::column_in_metadata(
      name.IsString() || name.IsNumber() ? name.ToString().Utf8Value() : null_value);
    metadata.column_metadata.push_back(column);
  }

  return std::move(cudf::io::orc_writer_options::builder(sink, table).metadata(&metadata).build());
}

}  // namespace

void Table::write_orc(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  CallbackArgs args{info};
  NODE_CUDF_EXPECT(args[1].IsObject(), "writeORC expects an Object of WriteORCOptions", env);

  std::string file_path = args[0];
  auto options          = args[1].As<Napi::Object>();

  cudf::table_view table = *this;
  cudf::io::write_orc(make_writer_options(options, cudf::io::sink_info{file_path}, table));
}

}  // namespace nv
