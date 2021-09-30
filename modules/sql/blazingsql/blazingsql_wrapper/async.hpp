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

#include <cudf/table/table.hpp>

#include <napi.h>

#include <memory>
#include <string>

namespace nv {
namespace blazingsql {

using SQLTaskCallback =
  typename std::function<std::pair<std::vector<std::string>, std::unique_ptr<cudf::table>>()>;

struct SQLTask : public Napi::AsyncWorker {
  SQLTask(Napi::Env const& env, SQLTaskCallback const& work);

  Napi::Promise run();

 protected:
  void Execute() override;
  void OnError(Napi::Error const& err) override;
  void OnOK() override;

  std::vector<napi_value> GetResult(Napi::Env env) override;

 private:
  bool queued_{false};

  SQLTaskCallback work_;
  std::vector<std::string> names_;
  std::unique_ptr<cudf::table> table_;

  Napi::Promise::Deferred deferred_;
};

}  // namespace blazingsql
}  // namespace nv
