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

#pragma once

#include <napi.h>

namespace nv {

class Task : public Napi::AsyncWorker {
 public:
  static inline Napi::Promise Rejected(const Napi::Value& value) {
    return (new Task(value.Env()))->Reject(value).Promise();
  }

  static inline Napi::Promise Resolved(const Napi::Value& value) {
    return (new Task(value.Env()))->Resolve(value).Promise();
  }

  // Safe to call from any thread
  static inline void Notify(void* task) { Task::Notify(static_cast<Task*>(task)); }

  static inline void Notify(Task* task) {
    if (!task->notified_ && (task->notified_ = true)) { task->Queue(); }
  }

  Task(Napi::Env env);
  Task(Napi::Env env, Napi::Value value);

  Napi::Promise Promise() const;
  Task& Inject(const Napi::Value& value);

  Task& Reject();
  Task& Reject(const Napi::Value& value);

  Task& Resolve();
  Task& Resolve(const Napi::Value& value);

  inline bool DelayResolve(const bool shouldDelayResolve) {
    if (!shouldDelayResolve) this->Resolve();
    return shouldDelayResolve;
  }

 protected:
  void Execute() override;
  void OnOK() override;

  bool settled_  = false;
  bool rejected_ = false;
  bool notified_ = false;

  Napi::Value val_;
  Napi::Reference<Napi::Value> ref_;
  Napi::Promise::Deferred deferred_;
};

}  // namespace nv
