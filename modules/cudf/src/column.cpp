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

#include <node_cudf/column.hpp>
#include <node_cudf/scalar.hpp>
#include <node_cudf/utilities/cpp_to_napi.hpp>
#include <node_cudf/utilities/dtypes.hpp>
#include <node_cudf/utilities/error.hpp>
#include <node_cudf/utilities/napi_to_cpp.hpp>

#include <node_cuda/utilities/cpp_to_napi.hpp>
#include <node_cuda/utilities/napi_to_cpp.hpp>

#include <node_rmm/device_buffer.hpp>
#include <node_rmm/utilities/napi_to_cpp.hpp>

#include <nv_node/macros.hpp>
#include <nv_node/utilities/args.hpp>
#include <nv_node/utilities/cpp_to_napi.hpp>
#include <nv_node/utilities/napi_to_cpp.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/types.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/traits.hpp>

#include <napi.h>

#include <memory>
#include <utility>

namespace nv {

namespace {

ObjectUnwrap<DeviceBuffer> device_buffer_from_memorylike(NapiToCPP const& value) {
  auto data = value;
  if (value.IsMemoryViewLike()) { data = value.ToObject().Get("buffer"); }
  if (DeviceBuffer::is_instance(data.val)) { return data.ToObject(); }
  return DeviceBuffer::New(data.operator Napi::ArrayBuffer());
}

ObjectUnwrap<DeviceBuffer> device_buffer_from_bool(NapiToCPP const& value, cudf::size_type size) {
  bool const valid = value;
  auto state       = valid ? cudf::mask_state::ALL_VALID : cudf::mask_state::ALL_NULL;
  return DeviceBuffer::New(
    std::make_unique<rmm::device_buffer>(cudf::create_null_mask(size, state)));
}

ObjectUnwrap<DeviceBuffer> null_mask_from_data_array(NapiToCPP const& value, cudf::size_type size) {
  auto const env       = value.Env();
  auto const vals      = value.As<Napi::Array>();
  auto const mask_size = cudf::bitmask_allocation_size_bytes(size);
  std::vector<cudf::bitmask_type> mask(mask_size / sizeof(cudf::bitmask_type), 0);
  auto const mask_data = mask.data();
  for (auto i = 0u, n = vals.Length(); i < n; ++i) {
    Napi::HandleScope scope{env};
    auto const elt = vals.Get(i);
    // Set the valid bit if the value isn't `null` or `undefined`
    if (!(elt.IsNull() or elt.IsUndefined())) { cudf::set_bit_unsafe(mask_data, i); }
  }
  return DeviceBuffer::New(mask_data, mask_size);
}

ObjectUnwrap<DeviceBuffer> null_mask_from_valid_array(NapiToCPP const& value,
                                                      cudf::size_type size) {
  auto const env       = value.Env();
  auto const vals      = value.As<Napi::Array>();
  auto const mask_size = cudf::bitmask_allocation_size_bytes(size);
  std::vector<cudf::bitmask_type> mask(mask_size / sizeof(cudf::bitmask_type), 0);
  auto const mask_data = mask.data();
  for (auto i = 0u, n = vals.Length(); i < n; ++i) {
    Napi::HandleScope scope{env};
    auto const elt = vals.Get(i);
    // Set the valid bit if the value is "truthy" by JS standards
    if (elt.ToBoolean().Value()) { cudf::set_bit_unsafe(mask_data, i); }
  }
  return DeviceBuffer::New(mask_data, mask_size);
}

ObjectUnwrap<DeviceBuffer> get_or_create_data(NapiToCPP const& value, cudf::data_type type) {
  if (value.IsMemoryLike()) { return device_buffer_from_memorylike(value); }
  if (value.IsArray()) {
    switch (type.id()) {
      case cudf::type_id::INT64: return DeviceBuffer::New<int64_t>(value.As<Napi::Array>());
      case cudf::type_id::UINT64: return DeviceBuffer::New<uint64_t>(value.As<Napi::Array>());
      default:
        auto buf = DeviceBuffer::New<double>(value.As<Napi::Array>());
        return (type.id() == cudf::type_id::FLOAT64) ? buf : [&]() {
          cudf::size_type size = buf->size() / sizeof(double);
          cudf::column_view view{cudf::data_type{cudf::type_id::FLOAT64}, size, buf->data()};
          return DeviceBuffer::New(std::move(cudf::cast(view, type)->release().data));
        }();
    }
  }
  return DeviceBuffer::New();
}

}  // namespace

//
// Public API
//

Napi::FunctionReference Column::constructor;

Napi::Object Column::Init(Napi::Env env, Napi::Object exports) {
  Napi::Function ctor =
    DefineClass(env,
                "Column",
                {
                  InstanceAccessor<&Column::type, &Column::type>("type"),
                  InstanceAccessor<&Column::data>("data"),
                  InstanceAccessor<&Column::null_mask>("mask"),
                  InstanceAccessor<&Column::offset>("offset"),
                  InstanceAccessor<&Column::size>("length"),
                  InstanceAccessor<&Column::has_nulls>("hasNulls"),
                  InstanceAccessor<&Column::null_count>("nullCount"),
                  InstanceAccessor<&Column::is_nullable>("nullable"),
                  InstanceAccessor<&Column::num_children>("numChildren"),

                  InstanceMethod<&Column::get_child>("getChild"),
                  InstanceMethod<&Column::get_value>("getValue"),
                  InstanceMethod<&Column::set_null_mask>("setNullMask"),
                  InstanceMethod<&Column::set_null_count>("setNullCount"),
                  // column/copying.cpp
                  InstanceMethod<&Column::gather>("gather"),
                  // column/binaryop.cpp
                  InstanceMethod<&Column::add>("add"),
                  InstanceMethod<&Column::sub>("sub"),
                  InstanceMethod<&Column::mul>("mul"),
                  InstanceMethod<&Column::div>("div"),
                  InstanceMethod<&Column::true_div>("true_div"),
                  InstanceMethod<&Column::floor_div>("floor_div"),
                  InstanceMethod<&Column::mod>("mod"),
                  InstanceMethod<&Column::pow>("pow"),
                  InstanceMethod<&Column::eq>("eq"),
                  InstanceMethod<&Column::ne>("ne"),
                  InstanceMethod<&Column::lt>("lt"),
                  InstanceMethod<&Column::gt>("gt"),
                  InstanceMethod<&Column::le>("le"),
                  InstanceMethod<&Column::ge>("ge"),
                  InstanceMethod<&Column::bitwise_and>("bitwise_and"),
                  InstanceMethod<&Column::bitwise_or>("bitwise_or"),
                  InstanceMethod<&Column::bitwise_xor>("bitwise_xor"),
                  InstanceMethod<&Column::logical_and>("logical_and"),
                  InstanceMethod<&Column::logical_or>("logical_or"),
                  InstanceMethod<&Column::coalesce>("coalesce"),
                  InstanceMethod<&Column::shift_left>("shift_left"),
                  InstanceMethod<&Column::shift_right>("shift_right"),
                  InstanceMethod<&Column::shift_right_unsigned>("shift_right_unsigned"),
                  InstanceMethod<&Column::log_base>("log_base"),
                  InstanceMethod<&Column::atan2>("atan2"),
                  InstanceMethod<&Column::null_equals>("null_equals"),
                  InstanceMethod<&Column::null_max>("null_max"),
                  InstanceMethod<&Column::null_min>("null_min"),
                  // column/stream_compaction.cpp
                  InstanceMethod<&Column::drop_nulls>("drop_nulls"),
                  InstanceMethod<&Column::drop_nans>("drop_nans"),
                  // column/transform.cpp
                  InstanceMethod<&Column::nans_to_nulls>("nans_to_nulls"),
                  // column/reduction.cpp
                  InstanceMethod<&Column::min>("min"),
                  InstanceMethod<&Column::max>("max"),
                  InstanceMethod<&Column::minmax>("minmax"),
                  InstanceMethod<&Column::sum>("sum"),
                  InstanceMethod<&Column::product>("product"),
                  InstanceMethod<&Column::any>("any"),
                  InstanceMethod<&Column::all>("all"),
                  InstanceMethod<&Column::sum_of_squares>("sum_of_squares"),
                  InstanceMethod<&Column::mean>("mean"),
                  InstanceMethod<&Column::median>("median"),
                  InstanceMethod<&Column::nunique>("nunique"),
                  InstanceMethod<&Column::variance>("var"),
                  InstanceMethod<&Column::std>("std"),
                  InstanceMethod<&Column::quantile>("quantile"),
                  // column/unaryop.cpp
                  InstanceMethod<&Column::cast>("cast"),
                  InstanceMethod<&Column::is_null>("isNull"),
                  InstanceMethod<&Column::is_valid>("isValid"),
                  InstanceMethod<&Column::is_nan>("isNaN"),
                  InstanceMethod<&Column::is_not_nan>("isNotNaN"),
                  InstanceMethod<&Column::sin>("sin"),
                  InstanceMethod<&Column::cos>("cos"),
                  InstanceMethod<&Column::tan>("tan"),
                  InstanceMethod<&Column::arcsin>("asin"),
                  InstanceMethod<&Column::arccos>("acos"),
                  InstanceMethod<&Column::arctan>("atan"),
                  InstanceMethod<&Column::sinh>("sinh"),
                  InstanceMethod<&Column::cosh>("cosh"),
                  InstanceMethod<&Column::tanh>("tanh"),
                  InstanceMethod<&Column::arcsinh>("asinh"),
                  InstanceMethod<&Column::arccosh>("acosh"),
                  InstanceMethod<&Column::arctanh>("atanh"),
                  InstanceMethod<&Column::exp>("exp"),
                  InstanceMethod<&Column::log>("log"),
                  InstanceMethod<&Column::sqrt>("sqrt"),
                  InstanceMethod<&Column::cbrt>("cbrt"),
                  InstanceMethod<&Column::ceil>("ceil"),
                  InstanceMethod<&Column::floor>("floor"),
                  InstanceMethod<&Column::abs>("abs"),
                  InstanceMethod<&Column::rint>("rint"),
                  InstanceMethod<&Column::bit_invert>("bit_invert"),
                  InstanceMethod<&Column::unary_not>("not"),
                });

  Column::constructor = Napi::Persistent(ctor);
  Column::constructor.SuppressDestruct();
  exports.Set("Column", ctor);

  return exports;
}

ObjectUnwrap<Column> Column::New(std::unique_ptr<cudf::column> column) {
  auto env   = constructor.Env();
  auto props = Napi::Object::New(env);

  props.Set("offset", 0);
  props.Set("length", column->size());
  props.Set("nullCount", column->null_count());
  props.Set("type", column_to_arrow_type(env, *column));

  auto contents = column->release();
  auto data     = std::move(contents.data);
  auto mask     = std::move(contents.null_mask);
  auto children = std::move(contents.children);

  props.Set("children", [&]() {
    auto ary = Napi::Array::New(env, children.size());
    for (size_t i = 0; i < children.size(); ++i) {  //
      ary.Set(i, New(std::move(children.at(i)))->Value());
    }
    return ary;
  }());

  props.Set("data", DeviceBuffer::New(std::move(data))->Value());
  props.Set("nullMask", DeviceBuffer::New(std::move(mask))->Value());

  return constructor.New({props});
}

Column::Column(CallbackArgs const& args) : Napi::ObjectWrap<Column>(args) {
  auto env = args.Env();

  NODE_CUDF_EXPECT(args.IsConstructCall(), "Column constructor requires 'new'", env);
  NODE_CUDF_EXPECT(args[0].IsObject(), "Column constructor requires a properties Object", env);

  NapiToCPP::Object props = args[0];

  NODE_CUDF_EXPECT(props.Has("type") && props.Get("type").IsObject(),
                   "Column constructor properties expects type to be an Object",
                   env);

  this->type_   = Napi::Persistent(props.Get("type").As<Napi::Object>());
  this->offset_ = props.Get("offset");

  this->children_ = Napi::Persistent(props.Has("children") ? props.Get("children").As<Napi::Array>()
                                                           : Napi::Array::New(Env(), 0));

  auto const data = get_or_create_data(props.Get("data"), type());
  this->data_     = data.reference();

  this->size_ = props.Get("length");

  if (this->size_ <= 0) {
    auto type = this->type();
    if (cudf::is_fixed_width(type)) {
      this->size_ = data->size() / cudf::size_of(type);
    } else if (type.id() == cudf::type_id::LIST) {
      if (num_children() > 0) { this->size_ = child(0).size() - 1; }
    } else if (type.id() == cudf::type_id::STRING) {
      if (num_children() > 0) { this->size_ = child(0).size() - 1; }
    } else if (type.id() == cudf::type_id::STRUCT) {
      if (num_children() > 0) {
        this->size_ = child(0).size();
        for (cudf::size_type i = 0; ++i < num_children();) {
          NODE_CUDF_EXPECT(
            child(i).size() == this->size_, "Struct column children must be the same size", env);
        }
      }
    }
    this->size_ -= this->offset_;
  }

  auto const mask = [&]() {
    // If "nullMask" was provided, use it to construct the validity bitmask
    if (props.Has("nullMask")) {
      auto const valid = props.Get("nullMask");
      if (valid.IsMemoryLike()) { return device_buffer_from_memorylike(valid); }
      if (valid.IsBoolean()) { return device_buffer_from_bool(valid, this->size_); }
      if (valid.IsArray()) { return null_mask_from_valid_array(valid, this->size_); }
    }
    // If "data" was provided as a JS Array, construct the valid bitmask from its non-null elements
    else if (props.Get("data").IsArray()) {
      return null_mask_from_data_array(props.Get("data"), this->size_);
    }
    // Otherwise return an empty bitmask indicating all-valid/non-nullable
    return DeviceBuffer::New();
  }();

  this->null_mask_ = mask.reference();
  if (!nullable()) {
    this->null_count_ = 0;
  } else if (!(props.Has("nullCount") && props.Get("nullCount").IsNumber())) {
    this->null_count_ = cudf::UNKNOWN_NULL_COUNT;
  } else {
    this->null_count_ =
      std::max(cudf::UNKNOWN_NULL_COUNT, static_cast<cudf::size_type>(props.Get("nullCount")));
  }
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
    null_mask_ = DeviceBuffer::New().reference();
  } else {
    ObjectUnwrap<DeviceBuffer> new_mask = new_null_mask;
    if (new_null_count > 0) {
      NODE_CUDF_EXPECT(new_mask->size() >= cudf::bitmask_allocation_size_bytes(this->size()),
                       "Column with null values must be nullable, and the null mask "
                       "buffer size should match the size of the column.");
    }
    null_mask_ = new_mask.reference();
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
  for (auto i = 0u; i < children.Length(); ++i) {
    auto child = children.Get(i).As<Napi::Object>();
    child_views.emplace_back(*Column::Unwrap(child));
  }

  return cudf::column_view{type,
                           size(),
                           data.data(),
                           static_cast<cudf::bitmask_type const*>(mask.data()),
                           null_count(),
                           offset(),
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
  for (auto i = 0u; i < children.Length(); ++i) {
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
                                   offset(),
                                   child_views};
}

Column::operator Napi::Value() const { return Value(); }

ObjectUnwrap<Column> Column::operator[](Column const& selection) const {
  if (selection.type().id() == cudf::type_id::BOOL8) {  //
    return this->apply_boolean_mask(selection);
  }
  return this->gather(selection);
}

//
// Private API
//

Napi::Value Column::type(Napi::CallbackInfo const& info) { return type_.Value(); }
void Column::type(Napi::CallbackInfo const& info, Napi::Value const& value) {
  type_ = Napi::Persistent(value.As<Napi::Object>());
}

Napi::Value Column::size(Napi::CallbackInfo const& info) { return CPPToNapi(info)(size()); }

Napi::Value Column::offset(Napi::CallbackInfo const& info) { return CPPToNapi(info)(offset()); }

Napi::Value Column::data(Napi::CallbackInfo const& info) { return data_.Value(); }

Napi::Value Column::has_nulls(Napi::CallbackInfo const& info) {
  return CPPToNapi(info)(null_count() > 0);
}

Napi::Value Column::is_nullable(Napi::CallbackInfo const& info) {
  return CPPToNapi(info)(nullable());
}

Napi::Value Column::num_children(Napi::CallbackInfo const& info) {
  return CPPToNapi(info)(num_children());
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
  if (!DeviceBuffer::is_instance(mask.val)) {
    mask = DeviceBuffer::New(mask.operator Span<char>())->Value();
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

Napi::Value Column::gather(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};
  if (!Column::is_instance(args[0])) {
    throw Napi::Error::New(info.Env(), "gather selection argument expects a Column");
  }
  auto& selection = *Column::Unwrap(args[0]);
  auto result     = (*this)[selection];
  return result->Value();
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
