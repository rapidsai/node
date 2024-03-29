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

#pragma once

#include <nv_node/utilities/args.hpp>

#include <cudf/io/parquet.hpp>

namespace nv {

cudf::io::table_input_metadata make_writer_columns_metadata(Napi::Object const& options,
                                                            cudf::table_view const& table);

Napi::Array get_output_names_from_metadata(Napi::Env const& env,
                                           cudf::io::table_with_metadata const& result);

Napi::Array get_output_cols_from_metadata(Napi::Env const& env,
                                          cudf::io::table_with_metadata const& result);

}  // namespace nv
