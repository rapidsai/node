// Copyright (c) 2020, NVIDIA CORPORATION.
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

#include "task.hpp"

namespace nv {

Task::Task(Napi::Env env) : Task(env, env.Undefined()) {}

Task::Task(Napi::Env env, Napi::Value value)
  : Napi::AsyncWorker(env), deferred_(Napi::Promise::Deferred::New(env)) {
  this->Inject(value);
}

Napi::Promise Task::Promise() const { return deferred_.Promise(); }

Task& Task::Inject(const Napi::Value& value) {
  if (!(value.IsObject() || value.IsFunction())) {
    val_ = value;
  } else {
    ref_.Reset(value);
  }
  return *this;
}

Task& Task::Reject() {
  if (notified_ == false && (rejected_ = true)) { Task::Notify(this); }
  return *this;
}

Task& Task::Reject(const Napi::Value& value) {
  return (notified_ ? *this : this->Inject(value)).Reject();
}

Task& Task::Resolve() {
  if (notified_ == false && !(rejected_ = false)) { Task::Notify(this); }
  return *this;
}

Task& Task::Resolve(const Napi::Value& value) {
  return (notified_ ? *this : this->Inject(value)).Resolve();
}

void Task::Execute() {}
void Task::OnOK() {
  if (!settled_ && (settled_ = true)) {
    Napi::HandleScope scope(Env());
    auto val = !ref_.IsEmpty() ? ref_.Value() : !val_.IsEmpty() ? val_ : Env().Undefined();
    rejected_ == true ? deferred_.Reject(val) : deferred_.Resolve(val);
  }
}

}  // namespace nv
