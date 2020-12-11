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

#include <node_cudf/column.hpp>
#include <node_cudf/scalar.hpp>
#include <node_cudf/utilities/cpp_to_napi.hpp>
#include <node_cudf/utilities/error.hpp>
#include <node_cudf/utilities/napi_to_cpp.hpp>

#include <node_cuda/utilities/cpp_to_napi.hpp>
#include <node_cuda/utilities/napi_to_cpp.hpp>

#include <node_rmm/device_buffer.hpp>
#include <node_rmm/utilities/napi_to_cpp.hpp>

#include <nv_node/macros.hpp>
#include <nv_node/utilities/args.hpp>
#include <nv_node/utilities/napi_to_cpp.hpp>

#include <cudf/binaryop.hpp>
#include <cudf/column/column.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/reduction.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>

#include <napi.h>

#include <memory>
#include <utility>

namespace nv {

//
// Public API
//

Napi::FunctionReference Column::constructor;

Napi::Object Column::Init(Napi::Env env, Napi::Object exports) {
  Napi::Function ctor =
    DefineClass(env,
                "Column",
                {
                  InstanceAccessor("type", &Column::type, nullptr, napi_enumerable),
                  InstanceAccessor("data", &Column::data, nullptr, napi_enumerable),
                  InstanceAccessor("mask", &Column::null_mask, nullptr, napi_enumerable),
                  InstanceAccessor("length", &Column::size, nullptr, napi_enumerable),
                  InstanceAccessor("hasNulls", &Column::has_nulls, nullptr, napi_enumerable),
                  InstanceAccessor("nullCount", &Column::null_count, nullptr, napi_enumerable),
                  InstanceAccessor("nullable", &Column::is_nullable, nullptr, napi_enumerable),
                  InstanceAccessor("numChildren", &Column::num_children, nullptr, napi_enumerable),
                  InstanceMethod("getChild", &Column::get_child),
                  InstanceMethod("getValue", &Column::get_value),
                  InstanceMethod("setNullMask", &Column::set_null_mask),
                  InstanceMethod("setNullCount", &Column::set_null_count),
                });

  Column::constructor = Napi::Persistent(ctor);
  Column::constructor.SuppressDestruct();
  exports.Set("Column", ctor);

  return exports;
}

Column Column::New(std::unique_ptr<cudf::column> column) {
  auto env   = constructor.Env();
  auto props = Napi::Object::New(env);

  props.Set("offset", 0);
  props.Set("length", column->size());
  props.Set("type", column->type().id());
  props.Set("nullCount", column->null_count());

  auto contents = std::move(column->release());
  auto data     = std::move(contents.data);
  auto mask     = std::move(contents.null_mask);
  auto children = std::move(contents.children);

  props.Set("children", [&]() {
    auto ary = Napi::Array::New(env, children.size());
    for (size_t i = 0; i < children.size(); ++i) {  //
      ary.Set(i, New(std::move(children.at(i))));
    }
    return ary;
  }());

  props.Set("data", DeviceBuffer::New(std::move(data)));
  props.Set("nullMask", DeviceBuffer::New(std::move(mask)));

  return std::move(*Unwrap(constructor.New({props})));
}

Column::Column(CallbackArgs const& args) : Napi::ObjectWrap<Column>(args) {
  NODE_CUDA_EXPECT(args.IsConstructCall(), "Column constructor requires 'new'");

  if (args.Length() != 1 || !args[0].IsObject()) { return; }

  Napi::Object props = args[0];

  cudf::size_type offset{props.Has("offset") ? NapiToCPP(props.Get("offset")) : 0};
  cudf::size_type length{props.Has("length") ? NapiToCPP(props.Get("length")) : 0};
  cudf::size_type null_count{props.Has("nullCount") ? NapiToCPP(props.Get("nullCount"))
                                                    : cudf::UNKNOWN_NULL_COUNT};
  cudf::data_type type{props.Has("type") ? NapiToCPP(props.Get("type")) : cudf::type_id::EMPTY};
  Napi::Array children = props.Has("children")  //
                           ? props.Get("children").As<Napi::Array>()
                           : Napi::Array::New(Env(), 0);

  DeviceBuffer data = std::move([&]() {
    if (cudf::is_fixed_width(type) && props.Has("data")) {
      auto data = NapiToCPP(props.Get("data"));
      if (data.IsMemoryLike()) {
        if (data.IsMemoryViewLike()) { data = NapiToCPP(data.ToObject().Get("buffer")); }
        if (DeviceBuffer::is_instance(data)) { return data.As<DeviceBuffer>(); }
        return DeviceBuffer::New(data.operator Span<char>());
      }
    }
    return DeviceBuffer::New();
  }());

  if (length == 0 && data.size() > 0 && cudf::is_fixed_width(type)) {
    length = data.size() / cudf::size_of(type);
  }

  DeviceBuffer mask = std::move([&]() {
    if (props.Has("nullMask")) {
      auto mask = NapiToCPP(props.Get("nullMask"));
      if (mask.IsMemoryLike()) {
        if (mask.IsMemoryViewLike()) { mask = NapiToCPP(mask.ToObject().Get("buffer")); }
        if (DeviceBuffer::is_instance(mask)) { return mask.As<DeviceBuffer>(); }
        return DeviceBuffer::New(mask.operator Span<char>());
      }
    }
    return DeviceBuffer::New();
  }());

  Initialize(data, length, DataType::New(type), mask, offset, null_count, children);
}

void Column::Initialize(DeviceBuffer const& data,
                        cudf::size_type size,
                        DataType const& type,
                        DeviceBuffer const& null_mask,
                        cudf::size_type offset,
                        cudf::size_type null_count,
                        Napi::Array const& children) {
  size_       = size;
  offset_     = offset;
  null_count_ = null_count;
  type_.Reset(type.Value(), 1);
  data_.Reset(data.Value(), 1);
  null_mask_.Reset(null_mask.Value(), 1);
  children_.Reset(children, 1);
}

void Column::Finalize(Napi::Env env) {
  data_.Reset();
  type_.Reset();
  null_mask_.Reset();
  children_.Reset();
}

// If the null count is known, return it. Else, compute and return it
cudf::size_type Column::null_count() const {
  CUDF_FUNC_RANGE();
  if (null_count_ <= cudf::UNKNOWN_NULL_COUNT) {
    auto& mask = this->null_mask();
    null_count_ =
      cudf::count_unset_bits(static_cast<cudf::bitmask_type const*>(mask.data()), 0, size());
  }
  return null_count_;
}

void Column::set_null_mask(Napi::Value const& new_null_mask, cudf::size_type new_null_count) {
  null_count_ = new_null_count;
  if (new_null_mask.IsNull() || new_null_mask.IsUndefined()) {
    null_mask_.Reset(DeviceBuffer::New().Value(), 1);
  } else {
    auto& new_mask = *DeviceBuffer::Unwrap(new_null_mask.ToObject());
    if (new_null_count > 0) {
      NODE_CUDF_EXPECT(new_mask.size() >= cudf::bitmask_allocation_size_bytes(this->size()),
                       "Column with null values must be nullable, and the null mask "
                       "buffer size should match the size of the column.");
    }
    null_mask_.Reset(new_mask.Value(), 1);
  }
}

void Column::set_null_count(cudf::size_type new_null_count) {
  if (new_null_count > 0) { NODE_CUDF_EXPECT(nullable(), "Invalid null count."); }
  null_count_ = new_null_count;
}

cudf::column_view Column::view() const {
  auto type     = this->type();
  auto& data    = this->data();
  auto& mask    = this->null_mask();
  auto children = children_.Value().As<Napi::Array>();

  // Create views of children
  std::vector<cudf::column_view> child_views;
  child_views.reserve(children.Length());
  for (auto i = 0; i < children.Length(); ++i) {
    auto child = children.Get(i).As<Napi::Object>();
    child_views.emplace_back(*Column::Unwrap(child));
  }

  return cudf::column_view{type,
                           size(),
                           data.data(),
                           static_cast<cudf::bitmask_type const*>(mask.data()),
                           null_count(),
                           0,
                           child_views};
}

cudf::mutable_column_view Column::mutable_view() {
  auto type     = this->type();
  auto& data    = this->data();
  auto& mask    = this->null_mask();
  auto children = children_.Value().As<Napi::Array>();

  // Create views of children
  std::vector<cudf::mutable_column_view> child_views;
  child_views.reserve(children.Length());
  for (auto i = 0; i < children.Length(); ++i) {
    auto child = children.Get(i).As<Napi::Object>();
    child_views.emplace_back(*Column::Unwrap(child));
  }

  // Store the old null count before resetting it. By accessing the value directly instead of
  // calling `null_count()`, we can avoid a potential invocation of `count_unset_bits()`. This does
  // however mean that calling `null_count()` on the resulting mutable view could still potentially
  // invoke `count_unset_bits()`.
  auto current_null_count = null_count_;

  // The elements of a column could be changed through a `mutable_column_view`, therefore the
  // existing `null_count` is no longer valid. Reset it to `UNKNOWN_NULL_COUNT` forcing it to be
  // recomputed on the next invocation of `null_count()`.
  set_null_count(cudf::UNKNOWN_NULL_COUNT);

  return cudf::mutable_column_view{type,
                                   size(),
                                   data.data(),
                                   static_cast<cudf::bitmask_type*>(mask.data()),
                                   current_null_count,
                                   0,
                                   child_views};
}

std::pair<Scalar, Scalar> Column::minmax() const {
  auto result = std::move(cudf::minmax(*this));
  return {Scalar::New(std::move(result.first)),  //
          Scalar::New(std::move(result.second))};
}

Column Column::operator==(Column const& other) const {
  return this->binary_operation<cudf::binary_operator::EQUAL, cudf::type_id::BOOL8>(other);
}

Column Column::operator==(Scalar const& other) const {
  return this->binary_operation<cudf::binary_operator::EQUAL, cudf::type_id::BOOL8>(other);
}

Column Column::operator<(Column const& other) const {
  return this->binary_operation<cudf::binary_operator::LESS, cudf::type_id::BOOL8>(other);
}

Column Column::operator<(Scalar const& other) const {
  return this->binary_operation<cudf::binary_operator::LESS, cudf::type_id::BOOL8>(other);
}

Column Column::operator<=(Column const& other) const {
  return this->binary_operation<cudf::binary_operator::LESS_EQUAL, cudf::type_id::BOOL8>(other);
}

Column Column::operator<=(Scalar const& other) const {
  return this->binary_operation<cudf::binary_operator::LESS_EQUAL, cudf::type_id::BOOL8>(other);
}

Column Column::operator>(Column const& other) const {
  return this->binary_operation<cudf::binary_operator::GREATER, cudf::type_id::BOOL8>(other);
}

Column Column::operator>(Scalar const& other) const {
  return this->binary_operation<cudf::binary_operator::GREATER, cudf::type_id::BOOL8>(other);
}

Column Column::operator>=(Column const& other) const {
  return this->binary_operation<cudf::binary_operator::GREATER_EQUAL, cudf::type_id::BOOL8>(other);
}

Column Column::operator>=(Scalar const& other) const {
  return this->binary_operation<cudf::binary_operator::GREATER_EQUAL, cudf::type_id::BOOL8>(other);
}

template <cudf::binary_operator op, cudf::type_id output_type>
Column Column::binary_operation(Column const& rhs) const {
  return Column::New(
    std::move(cudf::binary_operation(*this, rhs, op, cudf::data_type{output_type})));
}

template <cudf::binary_operator op, cudf::type_id output_type>
Column Column::binary_operation(Scalar const& rhs) const {
  return Column::New(
    std::move(cudf::binary_operation(*this, rhs, op, cudf::data_type{output_type})));
}

//
// Private API
//

Napi::Value Column::type(Napi::CallbackInfo const& info) { return type_.Value(); }

Napi::Value Column::size(Napi::CallbackInfo const& info) { return CPPToNapi(info)(size()); }

Napi::Value Column::data(Napi::CallbackInfo const& info) { return data_.Value(); }

Napi::Value Column::has_nulls(Napi::CallbackInfo const& info) {
  return CPPToNapi(info)(null_count() > 0);
}

Napi::Value Column::is_nullable(Napi::CallbackInfo const& info) {
  return CPPToNapi(info)(nullable());
}

Napi::Value Column::num_children(Napi::CallbackInfo const& info) {
  return CPPToNapi(info)(children_.Value().Length());
}

Napi::Value Column::null_mask(Napi::CallbackInfo const& info) { return null_mask_.Value(); }

Napi::Value Column::set_null_mask(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};

  auto mask            = args.Length() > 0 ? args[0] : NapiToCPP(info.Env().Null());
  cudf::size_type size = args.Length() > 1 ? args[1] : mask.IsNull() ? 0 : cudf::UNKNOWN_NULL_COUNT;

  NODE_CUDF_EXPECT((mask.IsNull() && size == 0) || mask.IsMemoryLike(),
                   "Expected nullMask to be an ArrayBuffer, ArrayBufferView, or DeviceBuffer",
                   Env());

  // Unwrap MemoryViews to get the buffer
  if (mask.IsMemoryViewLike()) { mask = NapiToCPP(mask.ToObject().Get("buffer")); }

  // If arg isn't a DeviceBuffer, copy the input data into a new DeviceBuffer
  if (!DeviceBuffer::is_instance(mask)) {
    mask = DeviceBuffer::New(mask.operator Span<char>()).Value();
  }

  set_null_mask(mask, size);

  return info.Env().Undefined();
}

Napi::Value Column::null_count(Napi::CallbackInfo const& info) {
  return CPPToNapi(info)(null_count());
}

Napi::Value Column::set_null_count(Napi::CallbackInfo const& info) {
  this->set_null_count(CallbackArgs{info}[0].operator cudf::size_type());
  return info.Env().Undefined();
}

Napi::Value Column::get_child(Napi::CallbackInfo const& info) {
  return children_.Value().Get(CallbackArgs{info}[0].operator cudf::size_type());
}

// Napi::Value Column::set_child(Napi::CallbackInfo const& info) {
// }

Napi::Value Column::get_value(Napi::CallbackInfo const& info) {
  return CPPToNapi(info)(cudf::get_element(*this, CallbackArgs{info}[0]));
}

// Napi::Value Column::set_value(Napi::CallbackInfo const& info) {
//   CallbackArgs args{info};
//   auto& scalar          = *Scalar::Unwrap(scalar_.Value());
//   cudf::size_type index = args[0];
//   scalar.set_value(info, info[1]);
// }

}  // namespace nv
