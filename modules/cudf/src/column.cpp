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

#include "column.hpp"
#include "cudf/types.hpp"
#include "macros.hpp"

#include <cstddef>
#include <nv_node/utilities/args.hpp>
#include <nv_node/utilities/cpp_to_napi.hpp>
#include <nv_node/utilities/napi_to_cpp.hpp>

#include <bits/stdint-intn.h>
#include <napi.h>
#include <iostream>
#include <map>
#include <string>

namespace nv {

Napi::FunctionReference Column::constructor;

Napi::Object Column::Init(Napi::Env env, Napi::Object exports) {
    Napi::Function ctor = DefineClass(
        env,
        "Column",
        {
          InstanceMethod("type", &Column::GetDataType),
          InstanceMethod("size", &Column::GetSize),
          InstanceMethod("set_null_mask", &Column::SetNullMask),
          InstanceMethod("set_null_count", &Column::SetNullCount),
          InstanceMethod("null_count", &Column::GetNullCount),
          InstanceMethod("nullable", &Column::Nullable),
          InstanceMethod("has_nulls", &Column::HasNulls),
        }
    );

    Column::constructor = Napi::Persistent(ctor);
    Column::constructor.SuppressDestruct();
    exports.Set("Column", ctor);

    return exports;
}

Napi::Value Column::New(
  cudf::data_type dtype, cudf::size_type size, rmm::device_buffer&& data,
  rmm::device_buffer&& null_mask, cudf::size_type null_count
) {

  auto buf = Column::constructor.New({});
  Column::Unwrap(buf)->column_.reset(new cudf::column{dtype, size, data, null_mask, null_count});

  return buf;
}

Column::Column(Napi::CallbackInfo const& info) : Napi::ObjectWrap<Column>(info) {
  CallbackArgs args{info};

  cudf::data_type dtype = cudf::data_type(
    static_cast<cudf::type_id>(static_cast<int32_t>(args[0]))
  );
  
  Span<char> data = args[2];
  cudf::size_type size = data.size();

  if(args.Length() == 3){
    this->column_.reset(
      new cudf::column{dtype, size, data}
    );
  } else if(args.Length() == 4){
    Span<char> null_mask = args[3];
    this->column_.reset(
      new cudf::column{dtype, size, data, null_mask}
    );
  } else if(args.Length() == 5){
    Span<char> null_mask = args[3];
    cudf::size_type null_count = args[4];
    this->column_.reset(
      new cudf::column{dtype, size, data, null_mask, null_count}
    );
  }

}

void Column::Finalize(Napi::Env env) {
  if (column_.get() != nullptr && column().size() > 0) {
    this->column_.reset(nullptr);
  }
  column_ = nullptr;
}

Napi::Value Column::GetDataType(Napi::CallbackInfo const& info) {
  return Napi::Number::New(info.Env(), static_cast<int32_t>(column().type().id()));
}

Napi::Value Column::GetSize(Napi::CallbackInfo const& info) {
  return CPPToNapi(info)(column().size());
}

Napi::Value Column::HasNulls(Napi::CallbackInfo const& info) {
  return CPPToNapi(info)(column().has_nulls());
}

Napi::Value Column::GetNullCount(Napi::CallbackInfo const& info) {
  return CPPToNapi(info)(column().null_count());
}

Napi::Value Column::Nullable(Napi::CallbackInfo const& info){
  return CPPToNapi(info)(column().nullable());
}

Napi::Value Column::SetNullCount(Napi::CallbackInfo const& info){
 CallbackArgs args{info};
 size_t new_null_count = args[0];
 column().set_null_count(new_null_count);
 return info.Env().Undefined();
}

Napi::Value Column::SetNullMask(Napi::CallbackInfo const& info){
  CallbackArgs args{info};
  
  Span<char> new_null_mask = args[0];
  if(args.Length() == 1){
    column().set_null_mask(static_cast<rmm::device_buffer>(new_null_mask));  
  }else{
    size_t new_null_count = args[1];
    column().set_null_mask(static_cast<rmm::device_buffer>(new_null_mask), new_null_count);
  }
  return info.Env().Undefined();
}

}
