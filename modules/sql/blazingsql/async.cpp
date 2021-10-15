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

#include "blazingsql_wrapper/async.hpp"

#include <node_cudf/table.hpp>

namespace nv {
namespace blazingsql {

SQLTask::SQLTask(Napi::Env const& env, SQLTaskCallback const& work)
  : AsyncWorker(env), work_(work), deferred_(Napi::Promise::Deferred::New(env)) {}

Napi::Promise SQLTask::run() {
  if (!queued_ && (queued_ = true)) { Queue(); }
  return deferred_.Promise();
}

void SQLTask::Execute() { std::tie(names_, tables_) = work_(); }

std::vector<napi_value> SQLTask::GetResult(Napi::Env env) {
  size_t i{0};
  auto names = Napi::Array::New(env, names_.size());
  std::for_each(names_.begin(), names_.end(), [&](auto const& name) mutable {  //
    names[i++] = name;
  });
  auto tables = Napi::Array::New(env, tables_.size());
  for (size_t i = 0; i < tables_.size(); ++i) {
    tables[i] = Table::New(env, std::move(tables_[i]));
  }
  return {names, tables};
}

void SQLTask::OnOK() {
  auto res      = GetResult(Env());
  auto obj      = Napi::Object::New(Env());
  obj["names"]  = res[0];
  obj["tables"] = res[1];
  deferred_.Resolve(obj);
}

void SQLTask::OnError(Napi::Error const& err) { deferred_.Reject(err.Value()); }

}  // namespace blazingsql
}  // namespace nv
