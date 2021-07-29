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

#include <nv_node/objectwrap.hpp>

#include <napi.h>

class Context;
namespace cudf {
class table;
}
namespace ral {
namespace cache {
struct graph;
}
}  // namespace ral

namespace nv {

struct ExecutionGraph : public EnvLocalObjectWrap<ExecutionGraph> {
  /**
   * @brief Initialize and export the ExecutionGraph JavaScript constructor and prototype.
   *
   * @param env The active JavaScript environment.
   * @param exports The exports object to decorate.
   * @return Napi::Function The ExecutionGraph constructor function.
   */
  static Napi::Function Init(Napi::Env env, Napi::Object exports);
  /**
   * @brief Construct a new ExecutionGraph instance from a ral::cache::graph.
   *
   * @param cache The shared pointer to the ExecutionGraph.
   */
  static wrapper_t New(Napi::Env const& env, std::shared_ptr<ral::cache::graph> graph);

  /**
   * @brief Construct a new ExecutionGraph instance from JavaScript.
   */
  ExecutionGraph(Napi::CallbackInfo const& info);

  inline operator std::shared_ptr<ral::cache::graph>() { return _graph; }

 private:
  std::shared_ptr<ral::cache::graph> _graph;

  bool _started{false};
  bool _results{false};
  Napi::Reference<Wrapper<Context>> _context;
  std::vector<std::string> _names{};
  std::vector<std::unique_ptr<cudf::table>> _tables{};

  Napi::Value start(Napi::CallbackInfo const& info);
  Napi::Value result(Napi::CallbackInfo const& info);
  Napi::Value send_to(Napi::CallbackInfo const& info);
};

}  // namespace nv
