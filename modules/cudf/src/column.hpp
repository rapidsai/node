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

#include <cudf/types.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <napi.h>

#include <cstddef>
#include <memory>
#include <string>
#include <utility>

namespace nv{

/**
 * @brief An owning wrapper around a cudf::Column.
 *
 */
class Column: public Napi::ObjectWrap<Column>{
public:
    /**
    * @brief Initialize and export the Column JavaScript constructor and prototype.
    *
    * @param env The active JavaScript environment.
    * @param exports The exports object to decorate.
    * @return Napi::Object The decorated exports object.
    */
    static Napi::Object Init(Napi::Env env, Napi::Object exports);

    /**
    * @brief Construct a new Column instance from C++ by deep copying the contents of other.
    *
    * @param other - The column to copy.
    */
    static Napi::Value New(
        cudf::column const& other
    );

    /**
    * @brief Construct a new Column instance from C++ by deep copying the contents of other.
    * Uses the specified stream and device_memory_resource for all allocations and copies
    *
    * @param other - The column to copy.
    * @param stream - CUDA stream used for device memory operations.
    * @param mr - Device memory resource to use for all device memory allocations.
    */
    static Napi::Value New(
        cudf::column const& other,
        cudaStream_t stream,
        rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()
    );

    /**
    * @brief Construct a new Column instance from existing device memory.
    *
    * @param dtype - The element type.
    * @param size - The number of elements in the column
    * @param data - The column's data
    * @param null_mask - Optional, column's null value indicator bitmask.
    * May be empty if null_count is 0 or UNKNOWN_NULL_COUNT
    * @param null_count - Optional, the count of null elements. If unknown, specify UNKNOWN_NULL_COUNT to
    * indicate that the null count should be computed on the first invocation of null_count()
    * @param children - Optional, vector of child columns
    */
    static Napi::Value New(
        cudf::data_type dtype, cudf::size_type size,
        rmm::device_buffer&& data,
        rmm::device_buffer&& null_mask = {},
        cudf::size_type null_count = cudf::UNKNOWN_NULL_COUNT,
        std::vector< std::unique_ptr< cudf::column >>&& children = {}
    );

    /**
    * @brief Construct a new Column instance from JavaScript.
    *
    */    
    Column(Napi::CallbackInfo const& info);

    /**
    * @brief Destructor called when the JavaScript VM garbage collects this Column
    * instance.
    *
    * @param env The active JavaScript environment.
    */
    void Finalize(Napi::Env env) override;

    /**
    * @brief Get underlying `cudf::column` object.
    *
    * @return cudf::column&
    */
    cudf::column& column() { return *column_; }

private:
    static Napi::FunctionReference constructor;

    Napi::Value type(Napi::CallbackInfo const& info);
    Napi::Value size(Napi::CallbackInfo const& info);
    Napi::Value hasNulls(Napi::CallbackInfo const& info);
    Napi::Value setNullCount(Napi::CallbackInfo const& info);
    Napi::Value nullCount(Napi::CallbackInfo const& info);
    Napi::Value nullable(Napi::CallbackInfo const& info);
    Napi::Value setNullMask(Napi::CallbackInfo const& info);
    Napi::Value child(Napi::CallbackInfo const& info);
    Napi::Value numChildren(Napi::CallbackInfo const& info);

    std::unique_ptr<cudf::column> column_;
};

} // namespace nv
