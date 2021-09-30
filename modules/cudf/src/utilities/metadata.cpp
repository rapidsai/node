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
