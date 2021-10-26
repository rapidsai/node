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

namespace nv {

cudf::io::table_input_metadata make_writer_columns_metadata(Napi::Object const& options,
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

  return metadata;
};

Napi::Array get_output_names_from_metadata(Napi::Env const& env,
                                           cudf::io::table_with_metadata const& result) {
  auto const& column_names = result.metadata.column_names;
  auto names               = Napi::Array::New(env, column_names.size());
  for (std::size_t i = 0; i < column_names.size(); ++i) { names.Set(i, column_names[i]); }
  return names;
}

Napi::Array get_output_cols_from_metadata(Napi::Env const& env,
                                          cudf::io::table_with_metadata const& result) {
  auto contents = result.tbl->release();
  auto columns  = Napi::Array::New(env, contents.size());
  for (std::size_t i = 0; i < contents.size(); ++i) {
    columns.Set(i, Column::New(env, std::move(contents[i]))->Value());
  }
  return columns;
}

}  // namespace nv
