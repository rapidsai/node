// Copyright (c) 2022, NVIDIA CORPORATION.
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
#include <node_cudf/utilities/napi_to_cpp.hpp>

#include <cudf/strings/split/partition.hpp>

namespace nv {

Napi::Value Column::string_partition(Napi::CallbackInfo const& info) {
  CallbackArgs const args{info};
  std::string const delimiter         = args[0];
  rmm::mr::device_memory_resource* mr = args[1];
  try {
    auto ary  = Napi::Array::New(info.Env(), 3);
    auto cols = cudf::strings::partition(view(), delimiter, mr)->release();
    for (std::size_t i = 0; i < cols.size(); ++i) {  //
      ary[i] = Column::New(Env(), std::move(cols[i]));
    }
    return ary;
  } catch (std::exception const& e) { throw Napi::Error::New(Env(), e.what()); }
}

}  // namespace nv
