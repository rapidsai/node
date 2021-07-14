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

#include "node_rmm/memory_resource.hpp"
#include "node_rmm/utilities/napi_to_cpp.hpp"

#include <node_cuda/device.hpp>

#include <thrust/optional.h>

namespace nv {

Napi::Function MemoryResource::Init(Napi::Env const& env, Napi::Object exports) {
  return DefineClass(
    env,
    "MemoryResource",
    {
      InstanceAccessor<&MemoryResource::get_device>("device"),
      InstanceAccessor<&MemoryResource::supports_streams>("supportsStreams"),
      InstanceAccessor<&MemoryResource::supports_get_mem_info>("supportsGetMemInfo"),
      InstanceMethod<&MemoryResource::is_equal>("isEqual"),
      InstanceMethod<&MemoryResource::get_mem_info>("getMemInfo"),
      InstanceMethod<&MemoryResource::add_bin>("addBin"),
      InstanceMethod<&MemoryResource::flush>("flush"),
      InstanceAccessor<&MemoryResource::get_file_path>("logFilePath"),
      InstanceAccessor<&MemoryResource::get_upstream_mr>("memoryResource"),
    });
}

MemoryResource::MemoryResource(CallbackArgs const& args)
  : EnvLocalObjectWrap<MemoryResource>(args) {
  auto env   = args.Env();
  auto& arg0 = args[0];
  auto& arg1 = args[1];
  auto& arg2 = args[2];
  auto& arg3 = args[3];

  NODE_CUDA_EXPECT(arg0.IsNumber(),
                   "MemoryResource constructor expects a numeric MemoryResourceType argument.",
                   args.Env());
  type_ = arg0;
  switch (type_) {
    case mr_type::cuda: {
      mr_ = std::make_shared<rmm::mr::cuda_memory_resource>();
      break;
    }

    case mr_type::managed: {
      mr_ = std::make_shared<rmm::mr::managed_memory_resource>();
      break;
    }

    case mr_type::pool: {
      NODE_CUDA_EXPECT(MemoryResource::IsInstance(arg1.val),
                       "PoolMemoryResource constructor expects an upstream MemoryResource from "
                       "which to allocate blocks for the pool.",
                       env);
      rmm::mr::device_memory_resource* mr = arg1;
      size_t const initial_pool_size      = arg2.IsNumber() ? arg2 : -1;
      size_t const maximum_pool_size      = arg3.IsNumber() ? arg3 : -1;
      upstream_mr_                        = Napi::Persistent(arg1.ToObject());
      mr_ = std::make_shared<rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource>>(
        mr,
        initial_pool_size == -1uL ? thrust::nullopt : thrust::make_optional(initial_pool_size),
        maximum_pool_size == -1uL ? thrust::nullopt : thrust::make_optional(maximum_pool_size));
      break;
    }

    case mr_type::fixed_size: {
      NODE_CUDA_EXPECT(MemoryResource::IsInstance(arg1.val),
                       "FixedSizeMemoryResource constructor expects an upstream MemoryResource "
                       "from which to allocate blocks for the pool.",
                       env);
      rmm::mr::device_memory_resource* mr = arg1;
      size_t const block_size             = arg2.IsNumber() ? arg2 : 1 << 20;
      size_t const blocks_to_preallocate  = arg3.IsNumber() ? arg3 : 128;
      upstream_mr_                        = Napi::Persistent(arg1.ToObject());
      mr_ = std::make_shared<rmm::mr::fixed_size_memory_resource<rmm::mr::device_memory_resource>>(
        mr, block_size, blocks_to_preallocate);
      break;
    }

    case mr_type::binning: {
      NODE_CUDA_EXPECT(MemoryResource::IsInstance(arg1.val),
                       "BinningMemoryResource constructor expects an upstream MemoryResource to "
                       "use for allocations larger than any of the bins.",
                       env);
      rmm::mr::device_memory_resource* mr = arg1;
      int8_t const min_size_exponent      = arg2.IsNumber() ? arg2 : -1;
      int8_t const max_size_exponent      = arg3.IsNumber() ? arg3 : -1;
      upstream_mr_                        = Napi::Persistent(arg1.ToObject());
      mr_ =
        (min_size_exponent <= -1 || max_size_exponent <= -1
           ? std::make_shared<rmm::mr::binning_memory_resource<rmm::mr::device_memory_resource>>(mr)
           : std::make_shared<rmm::mr::binning_memory_resource<rmm::mr::device_memory_resource>>(
               mr, min_size_exponent, max_size_exponent));
      break;
    }

    case mr_type::logging_adaptor: {
      NODE_CUDA_EXPECT(MemoryResource::IsInstance(arg1.val),
                       "LoggingResourceAdapter constructor expects an upstream MemoryResource.",
                       env);

      rmm::mr::device_memory_resource* mr = arg1;
      auto log_file_path                  = arg2.IsString() ? arg2.operator std::string() : "";
      bool auto_flush                     = arg3.IsBoolean() ? arg3 : false;

      if (log_file_path == "") {
        log_file_path = env.Global()
                          .Get("process")
                          .ToObject()
                          .Get("env")
                          .ToObject()
                          .Get("RMM_LOG_FILE")
                          .ToString();
      }

      NODE_CUDA_EXPECT(log_file_path != "",
                       "LoggingResourceAdapter constructor expects an RMM log file name string "
                       "argument or RMM_LOG_FILE environment variable",
                       env);

      upstream_mr_ = Napi::Persistent(arg1.ToObject());
      mr_ = std::make_shared<rmm::mr::logging_resource_adaptor<rmm::mr::device_memory_resource>>(
        mr, log_file_path, auto_flush);
      break;
    }
    default:
      throw Napi::Error::New(
        env,
        std::string{"Unknown MemoryResource type: "} + std::to_string(static_cast<uint8_t>(type_)));
  }
};

void MemoryResource::Finalize(Napi::Env env) {
  if (mr_ != nullptr) {
    Device::call_in_context(env, device().value(), [&] { mr_ = nullptr; });
  }
}

std::string MemoryResource::file_path() const { return log_file_path_; };

bool MemoryResource::is_equal(Napi::Env const& env,
                              rmm::mr::device_memory_resource const& other) const {
  return mr_->is_equal(other);
}

std::pair<std::size_t, std::size_t> MemoryResource::get_mem_info(
  Napi::Env const& env, rmm::cuda_stream_view stream) const {
  return mr_->get_mem_info(stream);
}

bool MemoryResource::supports_streams(Napi::Env const& env) const {
  return mr_->supports_streams();
}

bool MemoryResource::supports_get_mem_info(Napi::Env const& env) const {
  return mr_->supports_get_mem_info();
}

void MemoryResource::flush() {
  if (type_ == mr_type::logging_adaptor) { get_log_mr()->flush(); }
}

void MemoryResource::add_bin(size_t allocation_size) {
  if (type_ == mr_type::binning) { get_bin_mr()->add_bin(allocation_size); }
}

void MemoryResource::add_bin(size_t allocation_size, Napi::Object const& bin_resource) {
  if (type_ == mr_type::binning) {
    bin_mrs_.push_back(Napi::Persistent(bin_resource));
    get_bin_mr()->add_bin(allocation_size, *MemoryResource::Unwrap(bin_resource));
  }
}

void MemoryResource::flush(Napi::CallbackInfo const& info) {
  if (type_ == mr_type::logging_adaptor) { flush(); }
}

void MemoryResource::add_bin(Napi::CallbackInfo const& info) {
  if (type_ == mr_type::binning) {
    CallbackArgs const args{info};
    switch (info.Length()) {
      case 1: add_bin(args[0].operator size_t()); break;
      case 2: add_bin(args[0].operator size_t(), args[1]); break;
      default:
        NODE_CUDA_EXPECT(
          false, "add_bin expects numeric allocation_size and optional MemoryResource arguments.");
    }
  }
}

Napi::Value MemoryResource::is_equal(Napi::CallbackInfo const& info) {
  if (info.Length() != 1 || !IsInstance(info[0])) {  //
    return Napi::Value::From(info.Env(), false);
  }
  rmm::mr::device_memory_resource* other = CallbackArgs{info}[0];
  return Napi::Value::From(info.Env(), is_equal(info.Env(), *other));
}

Napi::Value MemoryResource::get_mem_info(Napi::CallbackInfo const& info) {
  auto env = info.Env();
  std::pair<std::size_t, std::size_t> mem_info{0, 0};
  if (supports_get_mem_info(env)) {
    mem_info =
      get_mem_info(env, info[0].IsNumber() ? CallbackArgs{info}[0] : rmm::cuda_stream_default);
  }
  return Napi::Value::From(info.Env(), mem_info);
}

Napi::Value MemoryResource::get_file_path(Napi::CallbackInfo const& info) {
  return Napi::Value::From(info.Env(), file_path());
}

Napi::Value MemoryResource::get_upstream_mr(Napi::CallbackInfo const& info) {
  return upstream_mr_.Value();
}

Napi::Value MemoryResource::supports_streams(Napi::CallbackInfo const& info) {
  return Napi::Value::From(info.Env(), supports_streams(info.Env()));
}

Napi::Value MemoryResource::supports_get_mem_info(Napi::CallbackInfo const& info) {
  return Napi::Value::From(info.Env(), supports_get_mem_info(info.Env()));
}

}  // namespace nv
