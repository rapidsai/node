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

#include "node_rmm/memory_resource.hpp"
#include "node_rmm/utilities/napi_to_cpp.hpp"

#include <thrust/optional.h>

namespace nv {

std::string const MemoryResource::class_path{"MemoryResource"};

Napi::Object MemoryResource::Init(Napi::Env env, Napi::Object exports) {
  exports.Set(
    "MemoryResource",
    DefineClass(env,
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
                }));

  return exports;
}

MemoryResource::MemoryResource(CallbackArgs const& args)
  : ObjectWrapMixin<MemoryResource>(), Napi::ObjectWrap<MemoryResource>(args) {
  auto& arg0 = args[0];
  auto& arg1 = args[1];
  auto& arg2 = args[2];
  auto& arg3 = args[3];

  NODE_CUDA_EXPECT(arg0.IsNumber(),
                   "MemoryResource constructor expects a numeric mr_type argument.");
  type_ = arg0;
  switch (type_) {
    case mr_type::cuda: {
      device_id_ = arg1.IsNumber() && static_cast<int32_t>(arg1) > -1 &&
                       static_cast<int32_t>(arg1) < Device::get_num_devices()
                     ? static_cast<int32_t>(arg1)
                     : Device::active_device_id();
      mr_.reset(rmm::mr::get_per_device_resource(rmm::cuda_device_id(device_id_)), [](auto* p) {});
      break;
    }

    case mr_type::managed: {
      device_id_ = Device::active_device_id();
      mr_.reset(new rmm::mr::managed_memory_resource());
      break;
    }

    case mr_type::pool: {
      NODE_CUDA_EXPECT(MemoryResource::is_instance(arg1.val),
                       "PoolMemoryResource constructor expects an upstream MemoryResource from "
                       "which to allocate blocks for the pool.");

      size_t const initial_pool_size = arg2.IsNumber() ? arg2 : -1;
      size_t const maximum_pool_size = arg3.IsNumber() ? arg3 : -1;
      upstream_mr_                   = Napi::Persistent(arg1.ToObject());
      mr_.reset(new rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource>(
        arg1,
        initial_pool_size == -1 ? thrust::nullopt : thrust::make_optional(initial_pool_size),
        maximum_pool_size == -1 ? thrust::nullopt : thrust::make_optional(maximum_pool_size)));
      break;
    }

    case mr_type::fixedsize: {
      NODE_CUDA_EXPECT(MemoryResource::is_instance(arg1.val),
                       "FixedSizeMemoryResource constructor expects an upstream MemoryResource "
                       "from which to allocate blocks for the pool.");
      rmm::mr::device_memory_resource* mr = arg1;
      size_t const block_size             = arg2.IsNumber() ? arg2 : 1 << 20;
      size_t const blocks_to_preallocate  = arg3.IsNumber() ? arg3 : 128;
      upstream_mr_                        = Napi::Persistent(arg1.ToObject());
      mr_.reset(new rmm::mr::fixed_size_memory_resource<rmm::mr::device_memory_resource>(
        mr, block_size, blocks_to_preallocate));
      break;
    }

    case mr_type::binning: {
      NODE_CUDA_EXPECT(MemoryResource::is_instance(arg1.val),
                       "BinningMemoryResource constructor expects an upstream MemoryResource to "
                       "use for allocations larger than any of the bins.");
      rmm::mr::device_memory_resource* mr = arg1;
      int8_t const min_size_exponent      = arg2.IsNumber() ? arg2 : -1;
      int8_t const max_size_exponent      = arg3.IsNumber() ? arg3 : -1;
      upstream_mr_                        = Napi::Persistent(arg1.ToObject());
      mr_.reset(min_size_exponent <= -1 || max_size_exponent <= -1
                  ? new rmm::mr::binning_memory_resource<rmm::mr::device_memory_resource>(mr)
                  : new rmm::mr::binning_memory_resource<rmm::mr::device_memory_resource>(
                      mr, min_size_exponent, max_size_exponent));
      break;
    }

    case mr_type::logging: {
      NODE_CUDA_EXPECT(MemoryResource::is_instance(arg1.val),
                       "LoggingResourceAdapter constructor expects an upstream MemoryResource.");

      std::string log_file_path = arg2.operator std::string();
      bool auto_flush           = arg3.IsBoolean() ? arg3 : false;

      if (log_file_path == "") {
        log_file_path = args.Env()
                          .Global()
                          .Get("process")
                          .ToObject()
                          .Get("env")
                          .ToObject()
                          .Get("RMM_LOG_FILE")
                          .ToString();
      }

      NODE_CUDA_EXPECT(log_file_path != "",
                       "LoggingResourceAdapter constructor expects an RMM log file name string "
                       "argument or RMM_LOG_FILE environment variable");

      upstream_mr_ = Napi::Persistent(arg1.ToObject());
      mr_.reset(new rmm::mr::logging_resource_adaptor<rmm::mr::device_memory_resource>(
        arg1, log_file_path, auto_flush));
      break;
    }
  }
};

ValueWrap<int32_t> MemoryResource::device() const {
  switch (type_) {
    case mr_type::cuda:
    case mr_type::managed: return {Env(), device_id_};
    default: return MemoryResource::Unwrap(upstream_mr_.Value())->device();
  }
}

ValueWrap<std::string const> MemoryResource::file_path() const { return {Env(), log_file_path_}; };

void MemoryResource::flush() {
  if (type_ == mr_type::logging) { get_log_mr()->flush(); }
}

void MemoryResource::add_bin(size_t allocation_size) {
  if (type_ == mr_type::binning) { get_bin_mr()->add_bin(allocation_size); }
}

void MemoryResource::add_bin(size_t allocation_size,
                             ObjectUnwrap<MemoryResource> const& bin_resource) {
  if (type_ == mr_type::binning) {
    bin_mrs_.push_back(bin_resource);
    get_bin_mr()->add_bin(allocation_size, bin_resource);
  }
}

Napi::Value MemoryResource::flush(Napi::CallbackInfo const& info) {
  if (type_ == mr_type::logging) { flush(); }
  return info.Env().Undefined();
}

Napi::Value MemoryResource::add_bin(Napi::CallbackInfo const& info) {
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
  return info.Env().Undefined();
}

ValueWrap<bool> MemoryResource::is_equal(
  rmm::mr::device_memory_resource const& other) const noexcept {
  return {Env(), mr_->is_equal(other)};
}

ValueWrap<std::pair<std::size_t, std::size_t>> MemoryResource::get_mem_info(
  rmm::cuda_stream_view stream) const {
  return {Env(), mr_->get_mem_info(stream)};
}

Napi::Value MemoryResource::get_device(Napi::CallbackInfo const& info) { return device(); }

Napi::Value MemoryResource::get_file_path(Napi::CallbackInfo const& info) { return file_path(); }

Napi::Value MemoryResource::get_upstream_mr(Napi::CallbackInfo const& info) {
  return upstream_mr_.Value();
}

ValueWrap<bool> MemoryResource::supports_streams() const noexcept {
  return {Env(), mr_->supports_streams()};
}

ValueWrap<bool> MemoryResource::supports_get_mem_info() const noexcept {
  return {Env(), mr_->supports_get_mem_info()};
}

}  // namespace nv
