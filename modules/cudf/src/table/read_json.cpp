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

#include <cudf/io/datasource.hpp>
#include <cudf/io/json.hpp>
#include <cudf/io/types.hpp>
#include <node_cudf/utilities/metadata.hpp>

namespace nv {

namespace {

cudf::io::json_reader_options make_reader_options(Napi::Object const& options,
                                                  cudf::io::source_info const& source) {
  auto env     = options.Env();
  auto is_null = [](Napi::Value const& val) {
    return val.IsNull() || val.IsEmpty() || val.IsUndefined();
  };
  auto has_opt  = [&](std::string const& key) { return options.Has(key); };
  auto napi_opt = [&](std::string const& key) -> Napi::Value {
    return has_opt(key) ? options.Get(key) : env.Undefined();
  };
  auto byte_opt = [&](std::string const& key) {
    return has_opt(key) ? options.Get(key).ToNumber() : 0;
  };
  auto bool_opt = [&](std::string const& key, bool default_val) {
    return has_opt(key) ? options.Get(key).ToBoolean() == true : default_val;
  };
  auto to_upper = [](std::string const& str) {
    std::string out = str;
    std::transform(
      str.begin(), str.end(), out.begin(), [](char const& c) { return std::toupper(c); });
    return out;
  };
  auto compression_type = [&](std::string const& key) {
    if (has_opt(key)) {
      auto type = to_upper(options.Get(key).ToString());
      if (type == "INFER") { return cudf::io::compression_type::AUTO; }
      if (type == "SNAPPY") { return cudf::io::compression_type::SNAPPY; }
      if (type == "GZIP") { return cudf::io::compression_type::GZIP; }
      if (type == "BZ2") { return cudf::io::compression_type::BZIP2; }
      if (type == "BROTLI") { return cudf::io::compression_type::BROTLI; }
      if (type == "ZIP") { return cudf::io::compression_type::ZIP; }
      if (type == "XZ") { return cudf::io::compression_type::XZ; }
    }
    return cudf::io::compression_type::NONE;
  };
  auto names_and_types = [&](std::string const& key) {
    std::vector<std::string> names;
    std::vector<cudf::data_type> types;
    auto dtypes = napi_opt(key);
    if (is_null(dtypes) || !dtypes.IsObject()) {
      names.resize(0);
      types.resize(0);
    } else {
      auto data_types = dtypes.ToObject();
      auto type_names = data_types.GetPropertyNames();
      names.reserve(type_names.Length());
      types.reserve(type_names.Length());
      for (uint32_t i = 0; i < type_names.Length(); ++i) {
        auto name  = type_names.Get(i).ToString().Utf8Value();
        auto _type = data_types.Get(name).As<Napi::Object>();
        auto type  = arrow_to_cudf_type(_type);
        names.push_back(name);
        types.push_back(type);
      }
    }
    names.shrink_to_fit();
    types.shrink_to_fit();
    return std::make_pair(std::move(names), std::move(types));
  };

  std::vector<std::string> names;
  std::vector<cudf::data_type> types;
  std::tie(names, types) = names_and_types("dataTypes");

  auto opts = std::move(cudf::io::json_reader_options::builder(source)
                          .dtypes(types)
                          .byte_range_offset(byte_opt("byteOffset"))
                          .byte_range_size(byte_opt("byteRange"))
                          .compression(compression_type("compression"))
                          .dayfirst(bool_opt("inferDatesWithDayFirst", false))
                          .lines(true)
                          .build());

  auto dt_columns = napi_opt("datetimeColumns");

  // Set the column names/dtypes and header inference flag
  if (names.size() > 0 && types.size() > 0) { opts.set_dtypes(types); }

  // set the column names or indices to infer as datetime columns
  if (!is_null(dt_columns) && dt_columns.IsArray()) {
    auto dt_cols = dt_columns.As<Napi::Array>();
    for (uint32_t i = 0; i < dt_cols.Length(); ++i) {
      std::vector<int32_t> date_indexes;
      std::vector<std::string> date_names;
      if (dt_cols.Get(i).IsString()) {
        date_names.push_back(dt_cols.Get(i).ToString());
      } else if (dt_cols.Get(i).IsNumber()) {
        date_indexes.push_back(dt_cols.Get(i).ToNumber());
      }
    }
  }

  return opts;
}

std::vector<cudf::io::host_buffer> get_host_buffers(std::vector<Span<char>> const& sources) {
  std::vector<cudf::io::host_buffer> buffers;
  buffers.reserve(sources.size());
  std::transform(sources.begin(), sources.end(), std::back_inserter(buffers), [&](auto const& buf) {
    return cudf::io::host_buffer{buf.data(), buf.size()};
  });
  return buffers;
}

Napi::Value read_json_files(Napi::Object const& options, std::vector<std::string> const& sources) {
  auto env    = options.Env();
  auto result = cudf::io::read_json(make_reader_options(options, cudf::io::source_info{sources}));
  auto output = Napi::Object::New(env);
  output.Set("names", get_output_names_from_metadata(env, result));
  output.Set("table", Table::New(env, get_output_cols_from_metadata(env, result)));
  return output;
}

Napi::Value read_json_strings(Napi::Object const& options, std::vector<Span<char>> const& sources) {
  auto env    = options.Env();
  auto result = cudf::io::read_json(
    make_reader_options(options, cudf::io::source_info{get_host_buffers(sources)}));
  auto output = Napi::Object::New(env);
  output.Set("names", get_output_names_from_metadata(env, result));
  output.Set("table", Table::New(env, get_output_cols_from_metadata(env, result)));
  return output;
}

}  // namespace

Napi::Value Table::read_json(Napi::CallbackInfo const& info) {
  auto env = info.Env();

  NODE_CUDF_EXPECT(info[0].IsObject(), "readCSV expects an Object of ReadCSVOptions", env);

  auto options = info[0].As<Napi::Object>();
  auto sources = options.Get("sources");

  NODE_CUDF_EXPECT(sources.IsArray(), "readCSV expects an Array of paths or buffers", env);
  try {
    return (options.Get("sourceType").ToString().Utf8Value() == "files")
             ? read_json_files(options, NapiToCPP{sources})
             : read_json_strings(options, NapiToCPP{sources});
  } catch (cudf::logic_error const& err) { NAPI_THROW(Napi::Error::New(env, err.what())); }
}

}  // namespace nv
