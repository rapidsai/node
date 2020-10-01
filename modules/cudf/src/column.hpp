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

#include <cudf/column/column.hpp>
#include <cudf/types.hpp>
#include <rmm/device_buffer.hpp>

#include <napi.h>

#include <cstddef>
#include <memory>
#include <string>
#include <utility>

namespace nv{

class Column: public Napi::ObjectWrap<Column>{
public:
    static Napi::Object Init(Napi::Env env, Napi::Object exports);

    //bindings for constructor - https://docs.rapids.ai/api/libcudf/stable/classcudf_1_1column.html#a3ba075937984a11d8cef416a85fd3089
    static Napi::Value New(
        cudf::data_type dtype, cudf::size_type size,
        rmm::device_buffer&& data,
        rmm::device_buffer&& null_mask = {},
        cudf::size_type null_count = cudf::UNKNOWN_NULL_COUNT
    );

    Column(Napi::CallbackInfo const& info);

    cudf::column& column() { return *column_; }
    std::string const& dtype() { return dtype_; }

    // cudf::size_type getSize() {return size_;}
    // void set_null_count(cudf::size_type val) {null_count_ = val;}
    // int null_count(){return null_count_;}
    // // bool has_nulls() {return null_count_ == 0;}
    // // cudf::size_type null_count() {return null_count();}

private:
    static Napi::FunctionReference constructor;

    Napi::Value GetDataType(Napi::CallbackInfo const& info);
    Napi::Value GetSize(Napi::CallbackInfo const& info);
    Napi::Value Release(Napi::CallbackInfo const& info);
    Napi::Value HasNulls(Napi::CallbackInfo const& info);
    Napi::Value SetNullCount(Napi::CallbackInfo const& info);
    Napi::Value GetNullCount(Napi::CallbackInfo const& info);
    // Napi::Value SetNullMask(Napi::CallbackInfo const& info);

    std::string dtype_;
    std::unique_ptr<cudf::column> column_;
};

} // namespace nv
