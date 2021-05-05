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

void check_join_args(std::string const& func_name, Napi::CallbackInfo const& info) {
  if (info.Length() != 3 and info.Length() != 4) {
    NAPI_THROW(Napi::Error::New(
      info.Env(),
      func_name + "expects a left, right, null_eqaulity and optionally a memory resource"));
  }

  if (!Table::IsInstance(info[0])) {
    throw Napi::Error::New(info.Env(), func_name + " left argument expects a Table");
  }

  if (!Table::IsInstance(info[1])) {
    throw Napi::Error::New(info.Env(), func_name + " right argument expects a Table");
  }

  if (!info[2].IsBoolean()) {
    throw Napi::Error::New(info.Env(), func_name + " null_equality argument expects an bool");
  }
}

Napi::Value make_gather_maps(
  Napi::Env const& env,
  std::pair<std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
            std::unique_ptr<rmm::device_uvector<cudf::size_type>>> gathers) {
  using namespace cudf;

  auto& lhs_map = gathers.first;
  auto& rhs_map = gathers.second;
  auto lhs_size = static_cast<size_type>(lhs_map->size());
  auto rhs_size = static_cast<size_type>(rhs_map->size());

  auto result = Napi::Array::New(env, 2);

  result.Set(
    0u,
    Column::New(env,
                std::make_unique<column>(data_type{type_id::INT32}, lhs_size, lhs_map->release())));

  result.Set(
    1u,
    Column::New(env,
                std::make_unique<column>(data_type{type_id::INT32}, rhs_size, rhs_map->release())));

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
  CallbackArgs const args{info};

  detail::check_join_args("full_join", info);

  Table::wrapper_t lhs                = args[0];
  Table::wrapper_t rhs                = args[1];
  bool null_equality                  = args[2];
  rmm::mr::device_memory_resource* mr = args[3];

  return detail::make_gather_maps(info.Env(),
                                  Table::full_join(info.Env(), lhs, rhs, null_equality, mr));
}

Napi::Value Table::inner_join(Napi::CallbackInfo const& info) {
  CallbackArgs const args{info};

  detail::check_join_args("inner_join", info);

  Table::wrapper_t lhs                = args[0];
  Table::wrapper_t rhs                = args[1];
  bool null_equality                  = args[2];
  rmm::mr::device_memory_resource* mr = args[3];

  return detail::make_gather_maps(info.Env(),
                                  Table::inner_join(info.Env(), lhs, rhs, null_equality, mr));
}

Napi::Value Table::left_join(Napi::CallbackInfo const& info) {
  CallbackArgs const args{info};

  detail::check_join_args("left_join", info);

  Table::wrapper_t lhs                = args[0];
  Table::wrapper_t rhs                = args[1];
  bool null_equality                  = args[2];
  rmm::mr::device_memory_resource* mr = args[3];

  return detail::make_gather_maps(info.Env(),
                                  Table::left_join(info.Env(), lhs, rhs, null_equality, mr));
}

Napi::Value Table::left_semi_join(Napi::CallbackInfo const& info) {
  using namespace cudf;

  CallbackArgs const args{info};

  detail::check_join_args("left_semi_join", info);

  Table::wrapper_t lhs                = args[0];
  Table::wrapper_t rhs                = args[1];
  bool null_equality                  = args[2];
  rmm::mr::device_memory_resource* mr = args[3];

  auto map      = Table::left_semi_join(info.Env(), lhs, rhs, null_equality, mr);
  auto map_size = static_cast<size_type>(map->size());

  return Column::New(info.Env(),
                     std::make_unique<column>(data_type{type_id::INT32}, map_size, map->release()));
}

Napi::Value Table::left_anti_join(Napi::CallbackInfo const& info) {
  using namespace cudf;

  CallbackArgs const args{info};

  detail::check_join_args("left_anti_join", info);

  Table::wrapper_t lhs                = args[0];
  Table::wrapper_t rhs                = args[1];
  bool null_equality                  = args[2];
  rmm::mr::device_memory_resource* mr = args[3];

  auto map      = Table::left_anti_join(info.Env(), lhs, rhs, null_equality, mr);
  auto map_size = static_cast<size_type>(map->size());

  return Column::New(info.Env(),
                     std::make_unique<column>(data_type{type_id::INT32}, map_size, map->release()));
}

}  // namespace nv
