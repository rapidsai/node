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

#include <node_cudf/utilities/error.hpp>

#include <nv_node/utilities/args.hpp>
#include <nv_node/utilities/cpp_to_napi.hpp>
#include <nv_node/utilities/wrap.hpp>

#include <cudf/scalar/scalar.hpp>
#include <cudf/types.hpp>

#include <napi.h>
#include <memory>

namespace nv {

class Scalar : public Napi::ObjectWrap<Scalar> {
 public:
  /**
   * @brief Initialize and export the Scalar JavaScript constructor and prototype.
   *
   * @param env The active JavaScript environment.
   * @param exports The exports object to decorate.
   * @return Napi::Object The decorated exports object.
   */
  static Napi::Object Init(Napi::Env env, Napi::Object exports);

  /**
   * @brief Construct a new Scalar instance from a scalar in device memory.
   *
   * @param scalar The scalar in device memory.
   */
  static ObjectUnwrap<Scalar> New(std::unique_ptr<cudf::scalar> scalar);

  /**
   * @brief Check whether an Napi value is an instance of `Scalar`.
   *
   * @param val The Napi::Value to test
   * @return true if the value is a `Scalar`
   * @return false if the value is not a `Scalar`
   */
  inline static bool is_instance(Napi::Value const& val) {
    return val.IsObject() and val.As<Napi::Object>().InstanceOf(constructor.Value());
  }

  /**
   * @brief Construct a new Scalar instance from JavaScript.
   *
   */
  Scalar(CallbackArgs const& args);

  /**
   * @brief Destructor called when the JavaScript VM garbage collects this Scalar instance.
   *
   * @param env The active JavaScript environment.
   */
  void Finalize(Napi::Env env) override;

  /**
   * @brief Move a unique scalar pointer so it's owned by this Scalar wrapper.
   *
   * @param other The scalar to move.
   */
  Scalar& operator=(std::unique_ptr<cudf::scalar>&& other) {
    scalar_ = std::move(other);
    return *this;
  }

  /**
   * @brief Returns the scalar's logical value type
   */
  cudf::data_type type() const noexcept { return scalar_->type(); }

  /**
   * @brief Updates the validity of the value
   *
   * @param is_valid true: set the value to valid. false: set it to null
   * @param stream CUDA stream used for device memory operations.
   */
  void set_valid(bool is_valid, cudaStream_t stream = 0) { scalar_->set_valid(is_valid, stream); }

  /**
   * @brief Indicates whether the scalar contains a valid value
   *
   * @note Using the value when `is_valid() == false` is undefined behaviour
   *
   * @param stream CUDA stream used for device memory operations.
   * @return true Value is valid
   * @return false Value is invalid/null
   */
  bool is_valid(cudaStream_t stream = 0) const { return scalar_->is_valid(stream); };

  template <typename scalar_type>
  inline operator scalar_type*() const {
    NODE_CUDF_EXPECT(cudf::type_to_id<typename scalar_type::value_type>() == type().id(),
                     "Invalid conversion from node_cudf::Scalar to cudf::scalar",
                     Env());
    return static_cast<scalar_type*>(scalar_.get());
  }

  operator cudf::scalar&() const;

  operator Napi::Value() const;

  Napi::Value get_value() const;

  void set_value(Napi::CallbackInfo const& info, Napi::Value const& value);

 private:
  static Napi::FunctionReference constructor;

  std::unique_ptr<cudf::scalar> scalar_;

  Napi::Value type(Napi::CallbackInfo const& info);
  Napi::Value get_value(Napi::CallbackInfo const& info);
};

}  // namespace nv
