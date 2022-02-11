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

#pragma once

#include <node_cudf/table.hpp>

#include <nv_node/objectwrap.hpp>
#include <nv_node/utilities/args.hpp>

#include <cudf/groupby.hpp>

#include <napi.h>

namespace nv {

/**
 * @brief An owning wrapper around a cudf::groupy::groupby.
 *
 */
struct GroupBy : public EnvLocalObjectWrap<GroupBy> {
  /**
   * @brief Initialize and export the Groupby JavaScript constructor and prototype.
   *
   * @param env The active JavaScript environment.
   * @param exports The exports object to decorate.
   * @return Napi::Function The GroupBy constructor function.
   */
  static Napi::Function Init(Napi::Env const& env, Napi::Object exports);

  /**
   * @brief Construct a new Groupby instance
   *
   */
  static wrapper_t New(Napi::Env const& env);

  /**
   * @brief Construct a new Groupby instance from JavaScript.
   *
   */
  GroupBy(CallbackArgs const& args);

 private:
  std::unique_ptr<cudf::groupby::groupby> groupby_;

  Napi::Value get_groups(Napi::CallbackInfo const& info);

  Napi::Value argmax(Napi::CallbackInfo const& info);
  Napi::Value argmin(Napi::CallbackInfo const& info);
  Napi::Value count(Napi::CallbackInfo const& info);
  Napi::Value max(Napi::CallbackInfo const& info);
  Napi::Value mean(Napi::CallbackInfo const& info);
  Napi::Value median(Napi::CallbackInfo const& info);
  Napi::Value min(Napi::CallbackInfo const& info);
  Napi::Value nth(Napi::CallbackInfo const& info);
  Napi::Value nunique(Napi::CallbackInfo const& info);
  Napi::Value std(Napi::CallbackInfo const& info);
  Napi::Value sum(Napi::CallbackInfo const& info);
  Napi::Value var(Napi::CallbackInfo const& info);
  Napi::Value quantile(Napi::CallbackInfo const& info);
  Napi::Value collect_list(Napi::CallbackInfo const& info);
  Napi::Value collect_set(Napi::CallbackInfo const& info);

  std::pair<Table::wrapper_t, rmm::mr::device_memory_resource*> _get_basic_args(
    Napi::CallbackInfo const& info);

  template <typename MakeAggregation>
  Napi::Value _single_aggregation(Napi::CallbackInfo const& info,
                                  Table::wrapper_t const& values_table,
                                  rmm::mr::device_memory_resource* const mr,
                                  MakeAggregation const& make_aggregation);
};

}  // namespace nv
