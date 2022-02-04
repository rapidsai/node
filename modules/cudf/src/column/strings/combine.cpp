// Copyright (c) 2021-2022, NVIDIA CORPORATION.
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
#include <node_cudf/scalar.hpp>
#include <node_cudf/table.hpp>

#include <node_rmm/device_buffer.hpp>

#include <cudf/column/column.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/combine.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/device_buffer.hpp>

#include <napi.h>

namespace nv {

Column::wrapper_t Column::concatenate(Napi::Env const& env,
                                      cudf::table_view const& columns,
                                      cudf::string_scalar const& separator,
                                      cudf::string_scalar const& narep,
                                      cudf::strings::separator_on_nulls separator_on_nulls,
                                      rmm::mr::device_memory_resource* mr) {
  try {
    return Column::New(
      env, cudf::strings::concatenate(columns, separator, narep, separator_on_nulls, mr));
  } catch (cudf::logic_error const& err) { NAPI_THROW(Napi::Error::New(env, err.what())); }
}

Napi::Value Column::concatenate(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};

  if (args.Length() != 5) {
    NAPI_THROW(Napi::Error::New(info.Env(),
                                "concatenate expects a columns, separator, narep, "
                                "separator_on_nulls and optionally a memory resource"));
  }

  Table::wrapper_t columns           = args[0];
  const std::string separator_string = args[1];
  const cudf::string_scalar separator{separator_string};

  auto narep = [&args]() {
    if (args[2].IsNull() or args[2].IsUndefined()) {
      return cudf::string_scalar{"", false};
    } else {
      const std::string& narep_string = args[2];
      return cudf::string_scalar{narep_string};
    }
  }();

  auto separator_on_nulls = [&args]() {
    bool separator_on_nulls_bool = args[3];
    return separator_on_nulls_bool ? cudf::strings::separator_on_nulls::YES
                                   : cudf::strings::separator_on_nulls::NO;
  }();

  rmm::mr::device_memory_resource* mr = args[4];

  return concatenate(info.Env(), columns->view(), separator, narep, separator_on_nulls, mr);
}

}  // namespace nv
