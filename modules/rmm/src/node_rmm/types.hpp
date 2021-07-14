// Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <cstdint>

namespace nv {

enum class mr_type : uint8_t {
  aligned_adaptor,
  arena,
  binning,
  cuda_async,
  cuda,
  device,
  fixed_size,
  limiting_adaptor,
  logging_adaptor,
  managed,
  polymorphic_allocator,
  pool,
  statistics_adaptor,
  thread_safe_adaptor,
  thrust_allocator_adaptor,
  tracking_adaptor,
};

}
