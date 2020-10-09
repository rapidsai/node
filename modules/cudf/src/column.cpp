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

#include "node_cudf/macros.hpp"
#include "utilities/napi_to_cpp.hpp"
//from node_rmm
#include "cuda_memory_resource.hpp" 

#include <cudf/column/column.hpp>
#include <cudf/types.hpp>
#include <nv_node/utilities/args.hpp>
#include <nv_node/utilities/cpp_to_napi.hpp>
#include <rmm/device_buffer.hpp>

#include <bits/stdint-intn.h>
#include <cstddef>
#include <memory>
#include <napi.h>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace nv {

//
// Public API
//

Napi::FunctionReference Column::constructor;

Napi::Object Column::Init(Napi::Env env, Napi::Object exports) {
    Napi::Function ctor = DefineClass(
        env,
        "Column",
        {
          InstanceMethod("type", &Column::type),
          InstanceMethod("size", &Column::size),
          InstanceMethod("setNullMask", &Column::setNullMask),
          InstanceMethod("setNullCount", &Column::setNullCount),
          InstanceMethod("nullCount", &Column::nullCount),
          InstanceMethod("nullable", &Column::nullable),
          InstanceMethod("hasNulls", &Column::hasNulls),
          InstanceMethod("child", &Column::child),
          InstanceMethod("numChildren", &Column::numChildren),
        }
    );

    Column::constructor = Napi::Persistent(ctor);
    Column::constructor.SuppressDestruct();
    exports.Set("Column", ctor);

    return exports;
}

Napi::Value Column::New(
  cudf::data_type dtype, cudf::size_type size, rmm::device_buffer&& data,
  rmm::device_buffer&& null_mask, cudf::size_type null_count,
  std::vector< std::unique_ptr< cudf::column >>&& children
) {

  auto buf = Column::constructor.New({});
  Column::Unwrap(buf)->column_.reset(
      new cudf::column(dtype, size, data, null_mask, null_count, std::move(children))
    );
  return buf;
}

Napi::Value Column::New(
  cudf::column const& column
) {

  auto buf = Column::constructor.New({});
  Column::Unwrap(buf)->column_.reset(
    new cudf::column(column)
  );
  return buf;
}


Napi::Value Column::New(
  cudf::column const& column,
  cudaStream_t stream,
  rmm::mr::device_memory_resource* mr
) {

  auto buf = Column::constructor.New({});
  Column::Unwrap(buf)->column_.reset(
    new cudf::column(column, stream, mr)
  );
  return buf;
}

Column::Column(Napi::CallbackInfo const& info) : Napi::ObjectWrap<Column>(info) {
  CallbackArgs args{info};
  
  if(args.Length() >= 1 && !info[0].IsNumber()){
    Column* col = Column::Unwrap(info[0].As<Napi::Object>());
    CudaMemoryResource* mr = {};
    
    if(args.Length() == 1){
      this->column_.reset(
        new cudf::column(col->column())
      );
    }
    else if(args.Length() == 2){
      cudaStream_t stream = args[1];
      this->column_.reset(
        new cudf::column(col->column(), stream)
      );
    }
    else if(args.Length() == 3){
      cudaStream_t stream = args[1];
      CudaMemoryResource* mr = CudaMemoryResource::Unwrap(info[2].As<Napi::Object>());
      this->column_.reset(
        new cudf::column(col->column(), stream, mr->Resource().get())
      );
    }
    

  }else if(args.Length() >= 3 ){
    
    cudf::data_type dtype = cudf::data_type(static_cast<cudf::type_id>(static_cast<int32_t>(args[0])));
    Span<char> data = args[2];
    cudf::size_type size = data.size();

    if(args.Length() == 3){
      this->column_.reset(
        new cudf::column(dtype, size, data)
      );
    }
    else if(args.Length() == 4){
      Span<char> null_mask = args[3];
      this->column_.reset(
        new cudf::column(dtype, size, data, null_mask)
      );
    } else if(args.Length() == 5){
      Span<char> null_mask = args[3];
      cudf::size_type null_count = args[4];
      this->column_.reset(
        new cudf::column(dtype, size, data, null_mask, null_count)
      );
    } else if(args.Length() == 6){
      Span<char> null_mask = args[3];
      cudf::size_type null_count = args[4];
      std::vector< std::unique_ptr< cudf::column >> children = args[5];
      this->column_.reset(
        new cudf::column(dtype, size, data, null_mask, null_count, std::move(children))
      );
    }

  }
  

}

void Column::Finalize(Napi::Env env) {
  if (column_.get() != nullptr && column().size() > 0) {
    this->column_.reset(nullptr);
  }
  column_ = nullptr;
}


//
// Private API
//

Napi::Value Column::type(Napi::CallbackInfo const& info) {
  return Napi::Number::New(info.Env(), static_cast<int32_t>(column().type().id()));
}

Napi::Value Column::size(Napi::CallbackInfo const& info) {
  return CPPToNapi(info)(column().size());
}

Napi::Value Column::hasNulls(Napi::CallbackInfo const& info) {
  return CPPToNapi(info)(column().has_nulls());
}

Napi::Value Column::nullCount(Napi::CallbackInfo const& info) {
  return CPPToNapi(info)(column().null_count());
}

Napi::Value Column::nullable(Napi::CallbackInfo const& info){
  return CPPToNapi(info)(column().nullable());
}

Napi::Value Column::setNullCount(Napi::CallbackInfo const& info){
 CallbackArgs args{info};
 size_t new_null_count = args[0];
 column().set_null_count(new_null_count);
 return info.Env().Undefined();
}

Napi::Value Column::setNullMask(Napi::CallbackInfo const& info){
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

Napi::Value Column::child(Napi::CallbackInfo const& info){
  CallbackArgs args{info};
  
  if(args.Length() == 1 && info[0].IsNumber()){
    if(static_cast<int32_t>(args[0]) < static_cast<int32_t>(column().num_children())){
      auto buf = Column::constructor.New({});
      Column::Unwrap(buf)->column_.reset(&column().child(args[0]));
      return buf; 
    }else{
      throw Napi::Error::New(info.Env(), "index out of range");
    }
  }
  throw Napi::Error::New(info.Env(), "invalid index type");
}

Napi::Value Column::numChildren(Napi::CallbackInfo const& info){
    return CPPToNapi(info)(column().num_children());
}

}
