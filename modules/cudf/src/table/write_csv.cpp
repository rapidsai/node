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

#include <node_cudf/column.hpp>
#include <node_cudf/table.hpp>

#include <cudf/io/csv.hpp>
#include <cudf/io/data_sink.hpp>
#include <cudf/io/types.hpp>

namespace nv {

namespace {

cudf::io::table_metadata make_writer_metadata(Napi::Object const& options,
                                              cudf::table_view const& table) {
  auto env      = options.Env();
  auto has_opt  = [&](std::string const& key) { return options.Has(key); };
  auto napi_opt = [&](std::string const& key) -> Napi::Value {
    return has_opt(key) ? options.Get(key) : env.Undefined();
  };
  auto str_opt = [&](std::string const& key, std::string const& default_val) {
    return has_opt(key) ? options.Get(key).ToString().Utf8Value() : default_val;
  };
  auto bool_opt = [&](std::string const& key, bool default_val) {
    return has_opt(key) ? options.Get(key).ToBoolean() == true : default_val;
  };
  cudf::io::table_metadata metadata{};
  auto null_value = str_opt("nullValue", "N/A");

  if (bool_opt("header", true)) {
    Napi::Array column_names = napi_opt("columnNames").IsArray()
                                 ? napi_opt("columnNames").As<Napi::Array>()
                                 : Napi::Array::New(env, table.num_columns());
    metadata.column_names.reserve(table.num_columns());
    for (auto i = 0u; i < column_names.Length(); ++i) {
      auto name = column_names.Has(i) ? column_names.Get(i) : env.Null();
      metadata.column_names.push_back(
        name.IsString() || name.IsNumber() ? name.ToString().Utf8Value() : null_value);
    }
  }
  return metadata;
}

cudf::io::csv_writer_options make_writer_options(Napi::Object const& options,
                                                 cudf::io::sink_info const& sink,
                                                 cudf::table_view const& table,
                                                 cudf::io::table_metadata* metadata) {
  auto has_opt = [&](std::string const& key) { return options.Has(key); };
  auto str_opt = [&](std::string const& key, std::string const& default_val) {
    return has_opt(key) ? options.Get(key).ToString().Utf8Value() : default_val;
  };
  auto long_opt = [&](std::string const& key, long default_val) {
    return has_opt(key) ? options.Get(key).ToNumber().Int32Value() : default_val;
  };
  auto bool_opt = [&](std::string const& key, bool default_val) {
    return has_opt(key) ? options.Get(key).ToBoolean() == true : default_val;
  };

  return std::move(cudf::io::csv_writer_options::builder(sink, table)
                     .metadata(metadata)
                     .na_rep(str_opt("nullValue", "N/A"))
                     .include_header(bool_opt("includeHeader", true))
                     .rows_per_chunk(long_opt("rowsPerChunk", 8))
                     .line_terminator(str_opt("lineTerminator", "\n"))
                     .inter_column_delimiter(str_opt("delimiter", ",")[0])
                     .true_value(str_opt("trueValue", "true"))
                     .false_value(str_opt("falseValue", "false"))
                     .build());
}

struct callback_sink : public cudf::io::data_sink {
  callback_sink(Napi::Function const& emit)
    : cudf::io::data_sink(),  //
      env_(emit.Env()),
      emit_(Napi::Persistent(emit)) {}

  size_t bytes_written() override { return bytes_written_; }

  void host_write(void const* data, size_t size) override {
    bytes_written_ += size;
    emit_({Napi::Buffer<char>::Copy(env_, static_cast<char const*>(data), size)});
  }

  void flush() override {}

 private:
  Napi::Env env_;
  size_t bytes_written_{0};
  Napi::FunctionReference emit_;
};

}  // namespace

Napi::Value Table::write_csv(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  NODE_CUDF_EXPECT(info[0].IsObject(), "writeCSV expects an Object of WriteCSVOptions", env);

  auto options = info[0].As<Napi::Object>();
  NODE_CUDF_EXPECT(options.Has("next"), "writeCSV expects a 'next' callback", env);

  auto emit = options.Get("next");
  NODE_CUDF_EXPECT(emit.IsFunction(), "writeCSV expects 'next' to be a function", env);

  cudf::table_view table = *this;
  callback_sink sink{emit.As<Napi::Function>()};
  auto metadata = make_writer_metadata(options, table);
  cudf::io::write_csv(make_writer_options(options, cudf::io::sink_info{&sink}, table, &metadata));
  return info.Env().Undefined();
}

}  // namespace nv
