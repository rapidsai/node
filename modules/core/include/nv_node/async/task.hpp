// Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

  Task(Napi::Env const& env) : Task(env, env.Undefined()){};
  Task(Napi::Env const& env, Napi::Value value)
    : Napi::AsyncWorker(env), deferred_(Napi::Promise::Deferred::New(env)) {
    this->Inject(value);
  }

  Napi::Promise Promise() const { return deferred_.Promise(); }

  Task& Inject(const Napi::Value& value) {
    if (!(value.IsObject() || value.IsFunction())) {
      val_ = value;
    } else {
      ref_.Reset(value);
    }
    return *this;
  }

  Task& Reject() {
    if (notified_ == false && (rejected_ = true)) { Task::Notify(this); }
    return *this;
  }
  Task& Reject(const Napi::Value& value) {
    return (notified_ ? *this : this->Inject(value)).Reject();
  }

  Task& Resolve() {
    if (notified_ == false && !(rejected_ = false)) { Task::Notify(this); }
    return *this;
  }
  Task& Resolve(const Napi::Value& value) {
    return (notified_ ? *this : this->Inject(value)).Resolve();
  }

  inline bool DelayResolve(const bool shouldDelayResolve) {
    if (!shouldDelayResolve) this->Resolve();
    return shouldDelayResolve;
  }

 protected:
  void Execute() override {}
  void OnOK() override {
    if (!settled_ && (settled_ = true)) {
      Napi::HandleScope scope(Env());
      auto val = !ref_.IsEmpty() ? ref_.Value() : !val_.IsEmpty() ? val_ : Env().Undefined();
      rejected_ == true ? deferred_.Reject(val) : deferred_.Resolve(val);
    }
  }

  bool settled_  = false;
  bool rejected_ = false;
  bool notified_ = false;

  Napi::Value val_;
  Napi::Reference<Napi::Value> ref_;
  Napi::Promise::Deferred deferred_;
};

}  // namespace nv
