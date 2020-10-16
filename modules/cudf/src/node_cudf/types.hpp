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
#include <cudf/types.hpp>
#include <nv_node/utilities/args.hpp>

namespace nv {

class DataType : public Napi::ObjectWrap<DataType> {
 public:
  /**
   * @brief Initialize and export the DataType JavaScript constructor and prototype.
   *
   * @param env The active JavaScript environment.
   * @param exports The exports object to decorate.
   * @return Napi::Object The decorated exports object.
   */
  static Napi::Object Init(Napi::Env env, Napi::Object exports);

  /**
   * @brief Construct a new DataType instance with a given type id.
   *
   * @param id The type's identifier.
   */
  static Napi::Object New(cudf::type_id id);

  /**
   * @brief Check whether an Napi value is an instance of `DataType`.
   *
   * @param val The Napi::Value to test
   * @return true if the value is a `DataType`
   * @return false if the value is not a `DataType`
   */
  inline static bool is_instance(Napi::Value const& val) {
    return val.IsObject() and val.As<Napi::Object>().InstanceOf(constructor.Value());
  }

  /**
   * @brief Construct a new DataType instance from JavaScript.
   *
   */
  DataType(CallbackArgs const& args);

  /**
   * @brief Initialize the DataType instance created by either C++ or JavaScript.
   *
   * @param id The type's identifier.
   */
  void Initialize(cudf::type_id id);

  operator cudf::data_type() const noexcept { return cudf::data_type{id_}; }

  cudf::type_id id() const noexcept { return id_; }

 private:
  static Napi::FunctionReference constructor;

  Napi::Value id(Napi::CallbackInfo const& info);

  cudf::type_id id_{cudf::type_id::EMPTY};
};

}  // namespace nv
