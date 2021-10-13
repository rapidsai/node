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
#include <node_cudf/utilities/buffer.hpp>
#include <node_cudf/utilities/cpp_to_napi.hpp>
#include <node_cudf/utilities/dtypes.hpp>

#include <cudf/detail/nvtx/ranges.hpp>

namespace nv {

Napi::Function Column::Init(Napi::Env const& env, Napi::Object exports) {
  return DefineClass(env,
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
                       InstanceMethod<&Column::copy>("copy"),
                       // column/filling.cpp
                       InstanceMethod<&Column::fill>("fill"),
                       InstanceMethod<&Column::fill_in_place>("fillInPlace"),
                       // column/binaryop.cpp
                       InstanceMethod<&Column::add>("add"),
                       InstanceMethod<&Column::sub>("sub"),
                       InstanceMethod<&Column::mul>("mul"),
                       InstanceMethod<&Column::div>("div"),
                       InstanceMethod<&Column::true_div>("trueDiv"),
                       InstanceMethod<&Column::floor_div>("floorDiv"),
                       InstanceMethod<&Column::mod>("mod"),
                       InstanceMethod<&Column::pow>("pow"),
                       InstanceMethod<&Column::eq>("eq"),
                       InstanceMethod<&Column::ne>("ne"),
                       InstanceMethod<&Column::lt>("lt"),
                       InstanceMethod<&Column::gt>("gt"),
                       InstanceMethod<&Column::le>("le"),
                       InstanceMethod<&Column::ge>("ge"),
                       InstanceMethod<&Column::bitwise_and>("bitwiseAnd"),
                       InstanceMethod<&Column::bitwise_or>("bitwiseOr"),
                       InstanceMethod<&Column::bitwise_xor>("bitwiseXor"),
                       InstanceMethod<&Column::logical_and>("logicalAnd"),
                       InstanceMethod<&Column::logical_or>("logicalOr"),
                       InstanceMethod<&Column::shift_left>("shiftLeft"),
                       InstanceMethod<&Column::shift_right>("shiftRight"),
                       InstanceMethod<&Column::shift_right_unsigned>("shiftRightUnsigned"),
                       InstanceMethod<&Column::log_base>("logBase"),
                       InstanceMethod<&Column::atan2>("atan2"),
                       InstanceMethod<&Column::null_equals>("nullEquals"),
                       InstanceMethod<&Column::null_max>("nullMax"),
                       InstanceMethod<&Column::null_min>("nullMin"),
                       // column/concatenate.cpp
                       InstanceMethod<&Column::concat>("concat"),
                       // column/stream_compaction.cpp
                       InstanceMethod<&Column::drop_nulls>("dropNulls"),
                       InstanceMethod<&Column::drop_nans>("dropNans"),
                       // column/filling.cpp
                       StaticMethod<&Column::sequence>("sequence"),
                       // column/transform.cpp
                       InstanceMethod<&Column::nans_to_nulls>("nansToNulls"),
                       // column/reduction.cpp
                       InstanceMethod<&Column::min>("min"),
                       InstanceMethod<&Column::max>("max"),
                       InstanceMethod<&Column::minmax>("minmax"),
                       InstanceMethod<&Column::sum>("sum"),
                       InstanceMethod<&Column::product>("product"),
                       InstanceMethod<&Column::any>("any"),
                       InstanceMethod<&Column::all>("all"),
                       InstanceMethod<&Column::sum_of_squares>("sumOfSquares"),
                       InstanceMethod<&Column::mean>("mean"),
                       InstanceMethod<&Column::median>("median"),
                       InstanceMethod<&Column::nunique>("nunique"),
                       InstanceMethod<&Column::variance>("var"),
                       InstanceMethod<&Column::std>("std"),
                       InstanceMethod<&Column::quantile>("quantile"),
                       InstanceMethod<&Column::cumulative_max>("cumulativeMax"),
                       InstanceMethod<&Column::cumulative_min>("cumulativeMin"),
                       InstanceMethod<&Column::cumulative_product>("cumulativeProduct"),
                       InstanceMethod<&Column::cumulative_sum>("cumulativeSum"),
                       // column/replacement.cpp
                       InstanceMethod<&Column::replace_nulls>("replaceNulls"),
                       InstanceMethod<&Column::replace_nans>("replaceNaNs"),
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
                       InstanceMethod<&Column::bit_invert>("bitInvert"),
                       InstanceMethod<&Column::unary_not>("not"),
                       // column/strings/attributes.cpp
                       InstanceMethod<&Column::count_bytes>("countBytes"),
                       InstanceMethod<&Column::count_characters>("countCharacters"),
                       // column/strings/combine.cpp
                       StaticMethod<&Column::concatenate>("concatenate"),
                       // column/strings/contains.cpp
                       InstanceMethod<&Column::contains_re>("containsRe"),
                       InstanceMethod<&Column::count_re>("countRe"),
                       InstanceMethod<&Column::matches_re>("matchesRe"),
                       // column/strings/json.cpp
                       InstanceMethod<&Column::get_json_object>("getJSONObject"),
                       // column/strings/padding.cpp
                       InstanceMethod<&Column::pad>("pad"),
                       InstanceMethod<&Column::zfill>("zfill"),
                       // column/convert.cpp
                       InstanceMethod<&Column::string_is_float>("stringIsFloat"),
                       InstanceMethod<&Column::strings_from_floats>("stringsFromFloats"),
                       InstanceMethod<&Column::strings_to_floats>("stringsToFloats"),
                       InstanceMethod<&Column::string_is_integer>("stringIsInteger"),
                       InstanceMethod<&Column::strings_from_integers>("stringsFromIntegers"),
                       InstanceMethod<&Column::strings_to_integers>("stringsToIntegers"),
                     });
}

Column::wrapper_t Column::New(Napi::Env const& env, std::unique_ptr<cudf::column> column) {
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
      ary.Set(i, New(env, std::move(children.at(i))));
    }
    return ary;
  }());

  props.Set("data", DeviceBuffer::New(env, std::move(data)));
  props.Set("nullMask", DeviceBuffer::New(env, std::move(mask)));

  return EnvLocalObjectWrap<Column>::New(env, {props});
}

Column::Column(CallbackArgs const& args) : EnvLocalObjectWrap<Column>(args) {
  auto env = args.Env();

  NODE_CUDF_EXPECT(args[0].IsObject(), "Column constructor expects an Object of properties", env);

  NapiToCPP::Object props = args[0];

  NODE_CUDF_EXPECT(props.Has("type") && props.Get("type").IsObject(),
                   "Column constructor properties expects a DataType Object",
                   env);

  offset_ = props.Has("offset") ? props.Get("offset") : 0;
  type_   = Napi::Persistent(props.Get("type").As<Napi::Object>());

  auto mask = Column::IsInstance(props) ? props.Get("mask").As<Napi::Value>()
              : props.Has("nullMask")   ? props.Get("nullMask").As<Napi::Value>()
                                        : env.Null();

  switch (type().id()) {
    case cudf::type_id::INT8:
    case cudf::type_id::INT16:
    case cudf::type_id::INT32:
    case cudf::type_id::INT64:
    case cudf::type_id::UINT8:
    case cudf::type_id::UINT16:
    case cudf::type_id::UINT32:
    case cudf::type_id::UINT64:
    case cudf::type_id::FLOAT64:
    case cudf::type_id::FLOAT32:
    case cudf::type_id::BOOL8:
    case cudf::type_id::TIMESTAMP_DAYS:
    case cudf::type_id::TIMESTAMP_SECONDS:
    case cudf::type_id::TIMESTAMP_MILLISECONDS:
    case cudf::type_id::TIMESTAMP_MICROSECONDS:
    case cudf::type_id::TIMESTAMP_NANOSECONDS: {
      children_ = Napi::Persistent(Napi::Array::New(env, 0));
      data_     = Napi::Persistent(data_to_devicebuffer(env, props.Get("data"), type()));
      size_ = std::max(0, cudf::size_type(data_.Value()->size() / cudf::size_of(type())) - offset_);
      null_mask_ =
        Napi::Persistent(mask.IsNull() ? data_to_null_bitmask(env, props.Get("data"), size_)
                                       : mask_to_null_bitmask(env, mask, size_));
      break;
    }
    case cudf::type_id::LIST:
    case cudf::type_id::STRING:
    case cudf::type_id::DICTIONARY32: {
      data_      = Napi::Persistent(DeviceBuffer::New(env));
      children_  = Napi::Persistent(props.Get("children").As<Napi::Array>());
      size_      = std::max(0, (num_children() > 0 ? child(0)->size() - 1 : 0) - offset_);
      null_mask_ = Napi::Persistent(mask_to_null_bitmask(env, mask, size_));
      break;
    }
    case cudf::type_id::STRUCT: {
      data_     = Napi::Persistent(DeviceBuffer::New(env));
      children_ = Napi::Persistent(props.Get("children").As<Napi::Array>());
      if (num_children() > 0) {
        size_ = std::max(0, child(0)->size() - offset_);
        for (cudf::size_type i = 0; ++i < num_children();) {
          NODE_CUDF_EXPECT(
            child(i)->size() == size_, "Struct column children must be the same size", env);
        }
      }
      null_mask_ = Napi::Persistent(mask_to_null_bitmask(env, mask, size_));
      break;
    }
    default: break;
  }

  size_ = props.Has("length") ? props.Get("length") : size_;

  set_null_count([&]() -> cudf::size_type {
    if (!nullable()) { return 0; }
    if (props.Has("nullCount")) {
      auto val = props.Get("nullCount");
      if (val.IsNumber()) {
        return std::max(cudf::UNKNOWN_NULL_COUNT, val.ToNumber().Int32Value());
      }
      if (val.IsBigInt()) {
        bool lossless{false};
        return std::max(cudf::UNKNOWN_NULL_COUNT,
                        static_cast<cudf::size_type>(val.As<Napi::BigInt>().Int64Value(&lossless)));
      }
    }
    return cudf::UNKNOWN_NULL_COUNT;
  }());
}

// If the null count is known, return it. Else, compute and return it
cudf::size_type Column::null_count() const {
  CUDF_FUNC_RANGE();
  if (!nullable()) {
    null_count_ = 0;
  } else if (null_count_ <= cudf::UNKNOWN_NULL_COUNT) {
    try {
      null_count_ = cudf::count_unset_bits(*null_mask(), 0, size_);
    } catch (std::exception const& e) {
      null_count_ = cudf::UNKNOWN_NULL_COUNT;
      NAPI_THROW(Napi::Error::New(Env(), e.what()));
    }
  }
  return null_count_;
}

void Column::set_null_mask(Napi::Value const& new_null_mask, cudf::size_type new_null_count) {
  if (new_null_mask.IsNull() or new_null_mask.IsUndefined() or !new_null_mask.IsObject()) {
    null_mask_ = Napi::Persistent(DeviceBuffer::New(new_null_mask.Env()));
    set_null_count(new_null_count);
  } else {
    DeviceBuffer::wrapper_t new_mask = new_null_mask.ToObject();
    if (new_null_count > 0) {
      NODE_CUDF_EXPECT(new_mask->size() >= cudf::bitmask_allocation_size_bytes(size_),
                       "Column with null values must be nullable, and the null mask "
                       "buffer size should match the size of the column.",
                       Env());
    }
    null_mask_ = Napi::Persistent(new_mask);
    set_null_count(new_null_count);
  }
}

void Column::set_null_count(cudf::size_type new_null_count) {
  NODE_CUDF_EXPECT(nullable() || new_null_count <= 0, "Invalid null count.", Env());
  null_count_ = std::max(std::min(size_, new_null_count), cudf::UNKNOWN_NULL_COUNT);
}

cudf::column_view Column::view() const {
  auto const type     = this->type();
  auto const data     = this->data();
  auto const mask     = this->null_mask();
  auto const children = children_.Value().As<Napi::Array>();

  // Create views of children
  std::vector<cudf::column_view> child_views;
  child_views.reserve(children.Length());
  for (auto i = 0u; i < children.Length(); ++i) {
    auto child = children.Get(i).As<Napi::Object>();
    child_views.emplace_back(*Column::Unwrap(child));
  }

  return cudf::column_view(type, size_, *data, *mask, null_count_, offset_, child_views);
}

cudf::mutable_column_view Column::mutable_view() {
  auto type     = this->type();
  auto data     = this->data();
  auto mask     = this->null_mask();
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

  return cudf::mutable_column_view{
    type, size_, *data, *mask, current_null_count, offset_, child_views};
}

Column::wrapper_t Column::operator[](Column const& selection) const {
  if (selection.type().id() == cudf::type_id::BOOL8) {  //
    return this->apply_boolean_mask(selection);
  }
  return this->gather(selection);
}

Napi::Value Column::type(Napi::CallbackInfo const& info) { return type_.Value(); }
void Column::type(Napi::CallbackInfo const& info, Napi::Value const& value) {
  type_ = Napi::Persistent(value.As<Napi::Object>());
}

Napi::Value Column::size(Napi::CallbackInfo const& info) {
  return Napi::Value::From(info.Env(), size_);
}

Napi::Value Column::offset(Napi::CallbackInfo const& info) {
  return Napi::Value::From(info.Env(), offset_);
}

Napi::Value Column::data(Napi::CallbackInfo const& info) { return data_.Value(); }

Napi::Value Column::has_nulls(Napi::CallbackInfo const& info) {
  return Napi::Value::From(info.Env(), null_count() > 0);
}

Napi::Value Column::null_count(Napi::CallbackInfo const& info) {
  return Napi::Value::From(info.Env(), null_count());
}

Napi::Value Column::is_nullable(Napi::CallbackInfo const& info) {
  return Napi::Value::From(info.Env(), nullable());
}

Napi::Value Column::num_children(Napi::CallbackInfo const& info) {
  return Napi::Value::From(info.Env(), num_children());
}

Napi::Value Column::null_mask(Napi::CallbackInfo const& info) { return null_mask_.Value(); }

void Column::set_null_mask(Napi::CallbackInfo const& info) {
  auto env  = info.Env();
  auto mask = mask_to_null_bitmask(env, info[0], size_);
  cudf::size_type null_count{info[0].IsNull() || info[0].IsUndefined() ? 0
                                                                       : cudf::UNKNOWN_NULL_COUNT};
  switch (info.Length()) {
    case 0: break;
    case 1: break;
    default:
      if (info[1].IsNumber()) {
        null_count = info[1].ToNumber().Int32Value();
      } else if (info[1].IsBigInt()) {
        bool lossless{false};
        null_count = info[1].As<Napi::BigInt>().Int64Value(&lossless);
      }
      break;
  }

  set_null_mask(mask, null_count);
}

void Column::set_null_count(Napi::CallbackInfo const& info) {
  this->set_null_count(info[0].ToNumber());
}

Napi::Value Column::gather(Napi::CallbackInfo const& info) {
  if (!Column::IsInstance(info[0])) {
    throw Napi::Error::New(info.Env(), "gather selection argument expects a Column");
  }
  return this->operator[](*Column::Unwrap(info[0].ToObject()));
}

Napi::Value Column::get_child(Napi::CallbackInfo const& info) {
  return children_.Value().Get(info[0].ToNumber());
}

Napi::Value Column::get_value(Napi::CallbackInfo const& info) {
  return Napi::Value::From(info.Env(), cudf::get_element(*this, info[0].ToNumber()));
}

}  // namespace nv
