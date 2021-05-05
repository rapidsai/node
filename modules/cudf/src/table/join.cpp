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
#include <node_cudf/utilities/napi_to_cpp.hpp>

#include <cudf/join.hpp>
#include <cudf/table/table_view.hpp>

namespace nv {

namespace detail {

void check_join_args(std::string func_name, CallbackArgs& args) {
  if (args.Length() != 3 and args.Length() != 4) {
    NAPI_THROW(Napi::Error::New(
      args.Env(),
      func_name + "expects a left, right, null_eqaulity and optionally a memory resource"));
  }

  if (!Table::is_instance(args[0])) {
    throw Napi::Error::New(args.Env(), func_name + " left argument expects a Table");
  }

  if (!Table::is_instance(args[1])) {
    throw Napi::Error::New(args.Env(), func_name + " right argument expects a Table");
  }

  if (!args[2].IsBoolean()) {
    throw Napi::Error::New(args.Env(), func_name + " null_equality argument expects an bool");
  }
}

Napi::Value prepare_gathers(
  Napi::Env const& env,
  std::pair<std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
            std::unique_ptr<rmm::device_uvector<cudf::size_type>>>& gathers) {
  using namespace cudf;

  auto left_gather = std::make_unique<column>(
    data_type(type_id::INT32), gathers.first->size(), gathers.first->release());

  auto right_gather = std::make_unique<column>(
    data_type(type_id::INT32), gathers.second->size(), gathers.second->release());

  auto result = Napi::Array::New(env, 2);
  result.Set(0u, Column::New(std::move(left_gather))->Value());
  result.Set(1u, Column::New(std::move(right_gather))->Value());

  return result;
}

}  // namespace detail

std::pair<std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
          std::unique_ptr<rmm::device_uvector<cudf::size_type>>>
Table::full_join(Napi::Env const& env,
                 Table const& left,
                 Table const& right,
                 bool null_equality,
                 rmm::mr::device_memory_resource* mr) {
  auto compare_nulls = null_equality ? cudf::null_equality::EQUAL : cudf::null_equality::UNEQUAL;
  try {
    return cudf::full_join(left, right, compare_nulls, mr);
  } catch (cudf::logic_error const& err) { NAPI_THROW(Napi::Error::New(env, err.what())); }
}

std::pair<std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
          std::unique_ptr<rmm::device_uvector<cudf::size_type>>>
Table::inner_join(Napi::Env const& env,
                  Table const& left,
                  Table const& right,
                  bool null_equality,
                  rmm::mr::device_memory_resource* mr) {
  auto compare_nulls = null_equality ? cudf::null_equality::EQUAL : cudf::null_equality::UNEQUAL;
  try {
    return cudf::inner_join(left, right, compare_nulls, mr);
  } catch (cudf::logic_error const& err) { NAPI_THROW(Napi::Error::New(env, err.what())); }
}

std::pair<std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
          std::unique_ptr<rmm::device_uvector<cudf::size_type>>>
Table::left_join(Napi::Env const& env,
                 Table const& left,
                 Table const& right,
                 bool null_equality,
                 rmm::mr::device_memory_resource* mr) {
  auto compare_nulls = null_equality ? cudf::null_equality::EQUAL : cudf::null_equality::UNEQUAL;
  try {
    return cudf::left_join(left, right, compare_nulls, mr);
  } catch (cudf::logic_error const& err) { NAPI_THROW(Napi::Error::New(env, err.what())); }
}

std::unique_ptr<rmm::device_uvector<cudf::size_type>> Table::left_semi_join(
  Napi::Env const& env,
  Table const& left,
  Table const& right,
  bool null_equality,
  rmm::mr::device_memory_resource* mr) {
  auto compare_nulls = null_equality ? cudf::null_equality::EQUAL : cudf::null_equality::UNEQUAL;
  try {
    return cudf::left_semi_join(left, right, compare_nulls, mr);
  } catch (cudf::logic_error const& err) { NAPI_THROW(Napi::Error::New(env, err.what())); }
}

std::unique_ptr<rmm::device_uvector<cudf::size_type>> Table::left_anti_join(
  Napi::Env const& env,
  Table const& left,
  Table const& right,
  bool null_equality,
  rmm::mr::device_memory_resource* mr) {
  auto compare_nulls = null_equality ? cudf::null_equality::EQUAL : cudf::null_equality::UNEQUAL;
  try {
    return cudf::left_anti_join(left, right, compare_nulls, mr);
  } catch (cudf::logic_error const& err) { NAPI_THROW(Napi::Error::New(env, err.what())); }
}

Napi::Value Table::full_join(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};

  detail::check_join_args("full_join", args);

  auto& left                          = *Table::Unwrap(args[0]);
  auto& right                         = *Table::Unwrap(args[1]);
  bool null_equality                  = args[2];
  rmm::mr::device_memory_resource* mr = args[3];

  auto gathers = Table::full_join(info.Env(), left, right, null_equality, mr);

  return detail::prepare_gathers(info.Env(), gathers);
}

Napi::Value Table::inner_join(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};

  detail::check_join_args("inner_join", args);

  auto& left                          = *Table::Unwrap(args[0]);
  auto& right                         = *Table::Unwrap(args[1]);
  bool null_equality                  = args[2];
  rmm::mr::device_memory_resource* mr = args[3];

  auto gathers = Table::inner_join(info.Env(), left, right, null_equality, mr);

  return detail::prepare_gathers(info.Env(), gathers);
}

Napi::Value Table::left_join(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};

  detail::check_join_args("left_join", args);

  auto& left                          = *Table::Unwrap(args[0]);
  auto& right                         = *Table::Unwrap(args[1]);
  bool null_equality                  = args[2];
  rmm::mr::device_memory_resource* mr = args[3];

  auto gathers = Table::left_join(info.Env(), left, right, null_equality, mr);

  return detail::prepare_gathers(info.Env(), gathers);
}

Napi::Value Table::left_semi_join(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};

  detail::check_join_args("left_semi_join", args);

  auto& left                          = *Table::Unwrap(args[0]);
  auto& right                         = *Table::Unwrap(args[1]);
  bool null_equality                  = args[2];
  rmm::mr::device_memory_resource* mr = args[3];

  auto gather = Table::left_semi_join(info.Env(), left, right, null_equality, mr);

  using namespace cudf;

  auto result =
    std::make_unique<column>(data_type(type_id::INT32), gather->size(), gather->release());

  return Column::New(std::move(result))->Value();
}

Napi::Value Table::left_anti_join(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};

  detail::check_join_args("left_anti_join", args);

  auto& left                          = *Table::Unwrap(args[0]);
  auto& right                         = *Table::Unwrap(args[1]);
  bool null_equality                  = args[2];
  rmm::mr::device_memory_resource* mr = args[3];

  auto gather = Table::left_anti_join(info.Env(), left, right, null_equality, mr);

  using namespace cudf;

  auto result =
    std::make_unique<column>(data_type(type_id::INT32), gather->size(), gather->release());

  return Column::New(std::move(result))->Value();
}

}  // namespace nv
