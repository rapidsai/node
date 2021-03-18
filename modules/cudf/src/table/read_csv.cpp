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
#include <cudf/io/datasource.hpp>
#include <cudf/io/types.hpp>

namespace nv {

namespace {

cudf::io::csv_reader_options make_reader_options(Napi::Object const& options,
                                                 cudf::io::source_info const& source) {
  auto env     = options.Env();
  auto is_null = [](Napi::Value const& val) {
    return val.IsNull() || val.IsEmpty() || val.IsUndefined();
  };
  auto has_opt  = [&](std::string const& key) { return options.Has(key); };
  auto napi_opt = [&](std::string const& key) -> Napi::Value {
    return has_opt(key) ? options.Get(key) : env.Undefined();
  };
  auto char_opt = [&](std::string const& key, std::string const& default_val) {
    return (has_opt(key) ? options.Get(key).ToString().Utf8Value() : default_val)[0];
  };
  auto long_opt = [&](std::string const& key) {
    return has_opt(key) ? options.Get(key).ToNumber().Int32Value() : -1;
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
  auto quote_style = [&](std::string const& key) {
    if (has_opt(key)) {
      auto style = to_upper(options.Get(key).ToString());
      if (style == "ALL") { return cudf::io::quote_style::ALL; }
      if (style == "NONE") { return cudf::io::quote_style::NONE; }
      if (style == "NONNUMERIC") { return cudf::io::quote_style::NONNUMERIC; }
    }
    return cudf::io::quote_style::MINIMAL;
  };
  auto names_and_types = [&](std::string const& key) {
    std::vector<std::string> names;
    std::vector<std::string> types;
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
        auto name = type_names.Get(i).ToString().Utf8Value();
        auto type = data_types.Get(name).ToString().Utf8Value();
        names.push_back(name);
        types.push_back(name + ":" + type);
      }
    }
    names.shrink_to_fit();
    types.shrink_to_fit();
    return std::make_pair(std::move(names), std::move(types));
  };

  std::vector<std::string> names;
  std::vector<std::string> types;
  std::tie(names, types) = names_and_types("dataTypes");

  auto opts = std::move(cudf::io::csv_reader_options::builder(source)
                          .byte_range_offset(0)
                          .byte_range_size(0)
                          .compression(compression_type("compression"))
                          .mangle_dupe_cols(bool_opt("renameDuplicateColumns", true))
                          .nrows(long_opt("numRows"))
                          .skiprows(long_opt("skipHead"))
                          .skipfooter(long_opt("skipTail"))
                          .quoting(quote_style("quoteStyle"))
                          .lineterminator(char_opt("lineTerminator", "\n"))
                          .quotechar(char_opt("quoteCharacter", "\""))
                          .decimal(char_opt("decimalCharacter", "."))
                          .delim_whitespace(bool_opt("whitespaceAsDelimiter", false))
                          .skipinitialspace(bool_opt("skipInitialSpaces", false))
                          .skip_blank_lines(bool_opt("skipBlankLines", true))
                          .doublequote(bool_opt("allowDoubleQuoting", true))
                          .keep_default_na(bool_opt("keepDefaultNA", true))
                          .na_filter(bool_opt("autoDetectNullValues", true))
                          .dayfirst(bool_opt("inferDatesWithDayFirst", false))
                          .delimiter(char_opt("delimiter", ","))
                          .thousands(char_opt("thousands", "\0"))
                          .comment(char_opt("comment", "\0"))
                          .build());

  auto header         = napi_opt("header");
  auto prefix         = napi_opt("prefix");
  auto null_values    = napi_opt("nullValues");
  auto true_values    = napi_opt("trueValues");
  auto false_values   = napi_opt("falseValues");
  auto dt_columns     = napi_opt("datetimeColumns");
  auto cols_to_return = napi_opt("columnsToReturn");

  // Set the column names/dtypes and header inference flag
  if (names.size() > 0 && types.size() > 0) {
    opts.set_names(names);
    opts.set_dtypes(types);
  }

  if (header.IsNumber()) {
    // Pass header row index down if provided
    opts.set_header(std::max(-1, header.ToNumber().Int32Value()));
  } else if (names.size() > 0 || header.IsNull()) {
    // * If header is `null` or names were explicitly provided, treat row 0 as data
    opts.set_header(-1);
  } else if (header.IsUndefined() ||
             (header.IsString() && header.ToString().Utf8Value() == "infer")) {
    // If header is `undefined` or "infer", try to parse row 0 as the header row
    opts.set_header(0);
  }

  // set the prefix
  if (!is_null(prefix) && prefix.IsString()) { opts.set_prefix(prefix.ToString().Utf8Value()); }
  // set the column names to return
  if (!is_null(cols_to_return) && cols_to_return.IsArray()) {
    auto const names_or_indices = cols_to_return.As<Napi::Array>();
    auto const string_col_names = [&]() {
      for (uint32_t i = 0; i < names_or_indices.Length(); ++i) {
        if (names_or_indices.Get(i).IsString()) { return true; }
      }
      return false;
    }();
    if (string_col_names) {
      opts.set_use_cols_names(NapiToCPP{names_or_indices});
    } else {
      opts.set_use_cols_indexes(NapiToCPP{names_or_indices});
    }
  }
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
      opts.set_infer_date_names(date_names);
      opts.set_infer_date_indexes(date_indexes);
    }
  }
  if (!is_null(null_values) && null_values.IsArray()) {
    opts.set_na_values(NapiToCPP{null_values});
  }
  if (!is_null(true_values) && true_values.IsArray()) {
    opts.set_true_values(NapiToCPP{true_values});
  }
  if (!is_null(false_values) && false_values.IsArray()) {
    opts.set_false_values(NapiToCPP{false_values});
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

Napi::Array get_output_names(Napi::Env const& env, cudf::io::table_with_metadata const& result) {
  auto const& column_names = result.metadata.column_names;
  auto names               = Napi::Array::New(env, column_names.size());
  for (std::size_t i = 0; i < column_names.size(); ++i) { names.Set(i, column_names[i]); }
  return names;
}

Napi::Array get_output_cols(Napi::Env const& env, cudf::io::table_with_metadata const& result) {
  auto contents = result.tbl->release();
  auto columns  = Napi::Array::New(env, contents.size());
  for (std::size_t i = 0; i < contents.size(); ++i) {
    columns.Set(i, Column::New(std::move(contents[i]))->Value());
  }
  return columns;
}

Napi::Value read_csv_files(Napi::Object const& options, std::vector<std::string> const& sources) {
  auto result = cudf::io::read_csv(make_reader_options(options, cudf::io::source_info{sources}));
  auto output = Napi::Object::New(options.Env());
  output.Set("names", get_output_names(options.Env(), result));
  output.Set("table", Table::New(get_output_cols(options.Env(), result)));
  return output;
}

Napi::Value read_csv_strings(Napi::Object const& options, std::vector<Span<char>> const& sources) {
  auto result = cudf::io::read_csv(
    make_reader_options(options, cudf::io::source_info{get_host_buffers(sources)}));
  auto output = Napi::Object::New(options.Env());
  output.Set("names", get_output_names(options.Env(), result));
  output.Set("table", Table::New(get_output_cols(options.Env(), result)));
  return output;
}

}  // namespace

Napi::Value Table::read_csv(Napi::CallbackInfo const& info) {
  auto env = info.Env();

  NODE_CUDF_EXPECT(info[0].IsObject(), "readCSV expects an Object of ReadCSVOptions", env);

  auto options = info[0].As<Napi::Object>();
  auto sources = options.Get("sources");

  NODE_CUDF_EXPECT(sources.IsArray(), "readCSV expects an Array of paths or buffers", env);
  try {
    return (options.Get("sourceType").ToString().Utf8Value() == "files")
             ? read_csv_files(options, NapiToCPP{sources})
             : read_csv_strings(options, NapiToCPP{sources});
  } catch (cudf::logic_error const& err) { NAPI_THROW(Napi::Error::New(env, err.what())); }
}

}  // namespace nv
