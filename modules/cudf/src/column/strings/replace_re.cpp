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

#include <node_rmm/memory_resource.hpp>

#include <cudf/strings/replace_re.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

namespace nv {

Column::wrapper_t Column::replace_re(std::string const& pattern,
                                     std::string const& replacement,
                                     cudf::size_type max_replace_count,
                                     cudf::strings::regex_flags const flags,
                                     rmm::mr::device_memory_resource* mr) const {
  try {
    return Column::New(Env(),
                       cudf::strings::replace_re(
                         this->view(),
                         pattern,
                         replacement,
                         max_replace_count < 0 ? std::nullopt : std::optional{max_replace_count},
                         flags,
                         mr));
  } catch (std::exception const& e) { throw Napi::Error::New(Env(), e.what()); }
}

Napi::Value Column::replace_re(Napi::CallbackInfo const& info) {
  CallbackArgs const args{info};

  std::string const& pattern     = args[0];
  std::string const& replacement = args[1];
  cudf::size_type const& count   = args[2];
  Napi::Object const& options    = args[3];

  uint32_t flags{cudf::strings::regex_flags::DEFAULT};

  if (options.Get("dotAll").ToBoolean()) { flags |= cudf::strings::regex_flags::DOTALL; }
  if (options.Get("multiline").ToBoolean()) { flags |= cudf::strings::regex_flags::MULTILINE; }

  return replace_re(pattern, replacement, count, cudf::strings::regex_flags{flags}, args[4]);
}

}  // namespace nv
