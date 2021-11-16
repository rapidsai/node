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

#include <cudf/io/orc.hpp>
#include <node_cudf/table.hpp>
#include <node_cudf/utilities/metadata.hpp>

namespace nv {

namespace {

cudf::io::orc_reader_options make_reader_options(Napi::Object const& options,
                                                 cudf::io::source_info const& source) {
  auto env     = options.Env();
  auto is_null = [](Napi::Value const& val) {
    return val.IsNull() || val.IsEmpty() || val.IsUndefined();
  };
  auto has_opt  = [&](std::string const& key) { return options.Has(key); };
  auto napi_opt = [&](std::string const& key) -> Napi::Value {
    return has_opt(key) ? options.Get(key) : env.Undefined();
  };
  auto long_opt = [&](std::string const& key) {
    return has_opt(key) ? options.Get(key).ToNumber().Int32Value() : -1;
  };
  auto bool_opt = [&](std::string const& key, bool default_val) {
    return has_opt(key) ? options.Get(key).ToBoolean() == true : default_val;
  };

  auto napi_stripes = napi_opt("stripes");
  std::vector<std::vector<cudf::size_type>> stripes;
  if (!is_null(napi_stripes) && napi_stripes.IsArray()) {
    auto arr = napi_stripes.As<Napi::Array>();
    for (size_t i = 0; i < arr.Length(); ++i) { stripes.push_back(NapiToCPP{arr.Get(i)}); }
  }

  auto opts = std::move(cudf::io::orc_reader_options::builder(source)
                          .num_rows(long_opt("numRows"))
                          .use_index(bool_opt("useIndex", true))
                          .build());

  // These cannot be both set together (cudf exception), so we only set them depending on if
  // the options contains a definition for them.
  if (!stripes.empty()) { opts.set_stripes(stripes); }
  if (has_opt("skipRows")) { opts.set_skip_rows(long_opt("skipRows")); }

  auto decimal_cols_as_floats = napi_opt("decimalColsAsFloats");
  if (!is_null(decimal_cols_as_floats) && decimal_cols_as_floats.IsArray()) {
    opts.set_decimal_cols_as_float(NapiToCPP{decimal_cols_as_floats});
  }

  auto columns = napi_opt("columns");
  if (!is_null(columns) && columns.IsArray()) { opts.set_columns(NapiToCPP{columns}); }

  return opts;
}

Napi::Value read_orc_files(Napi::Object const& options, std::vector<std::string> const& sources) {
  auto env    = options.Env();
  auto result = cudf::io::read_orc(make_reader_options(options, cudf::io::source_info{sources}));
  auto output = Napi::Object::New(env);
  output.Set("names", get_output_names_from_metadata(env, result));
  output.Set("table", Table::New(env, get_output_cols_from_metadata(env, result)));
  return output;
}

std::vector<cudf::io::host_buffer> get_host_buffers(std::vector<Span<uint8_t>> const& sources) {
  std::vector<cudf::io::host_buffer> buffers;
  buffers.reserve(sources.size());
  std::transform(sources.begin(), sources.end(), std::back_inserter(buffers), [&](auto const& buf) {
    return cudf::io::host_buffer{static_cast<Span<char>>(buf), buf.size()};
  });
  return buffers;
}

Napi::Value read_orc_sources(Napi::Object const& options,
                             std::vector<Span<uint8_t>> const& sources) {
  auto env    = options.Env();
  auto result = cudf::io::read_orc(
    make_reader_options(options, cudf::io::source_info{get_host_buffers(sources)}));
  auto output = Napi::Object::New(env);
  output.Set("names", get_output_names_from_metadata(env, result));
  output.Set("table", Table::New(env, get_output_cols_from_metadata(env, result)));
  return output;
}

}  // namespace

Napi::Value Table::read_orc(Napi::CallbackInfo const& info) {
  auto env = info.Env();

  NODE_CUDF_EXPECT(info[0].IsObject(), "readORC expects an Object of ReadORCOptions", env);

  auto options = info[0].As<Napi::Object>();
  auto sources = options.Get("sources");

  NODE_CUDF_EXPECT(sources.IsArray(), "readORC expects an Array of paths or buffers", env);
  try {
    return (options.Get("sourceType").ToString().Utf8Value() == "files")
             ? read_orc_files(options, NapiToCPP{sources})
             : read_orc_sources(options, NapiToCPP{sources});
  } catch (cudf::logic_error const& err) { NAPI_THROW(Napi::Error::New(env, err.what())); }
}

}  // namespace nv
