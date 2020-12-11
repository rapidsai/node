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

#include <node_rmm/memory_resource.hpp>
#include <node_rmm/utilities/cpp_to_napi.hpp>
#include <node_rmm/utilities/napi_to_cpp.hpp>

#include <node_cuda/utilities/cpp_to_napi.hpp>
#include <node_cuda/utilities/error.hpp>

#include <nv_node/utilities/args.hpp>

#include <thrust/optional.h>

#include <napi.h>
#include <memory>

namespace nv {

// MemoryResource

bool MemoryResource::is_instance(Napi::Value const& val) {
  return val.IsObject() and
         (CudaMemoryResource::is_instance(val) or ManagedMemoryResource::is_instance(val) or
          PoolMemoryResource::is_instance(val) or FixedSizeMemoryResource::is_instance(val) or
          BinningMemoryResource::is_instance(val) or LoggingResourceAdapter::is_instance(val));
}

Napi::Value MemoryResource::is_equal(Napi::CallbackInfo const& info) {
  if (info.Length() != 1 || !is_instance(info[0])) { return CPPToNapi(info)(false); }
  rmm::mr::device_memory_resource* other = CallbackArgs{info}[0];
  return CPPToNapi(info)(is_equal(*other));
}

Napi::Value MemoryResource::get_mem_info(Napi::CallbackInfo const& info) {
  if (supports_get_mem_info()) {
    return CPPToNapi(info)(
      get_mem_info(info.Length() != 1 ? rmm::cuda_stream_default : CallbackArgs{info}[0]));
  }
  return CPPToNapi(info)(std::make_pair<std::size_t, std::size_t>(0, 0));
}

Napi::Value MemoryResource::supports_streams(Napi::CallbackInfo const& info) {
  return CPPToNapi(info)(supports_streams());
}

Napi::Value MemoryResource::supports_get_mem_info(Napi::CallbackInfo const& info) {
  return CPPToNapi(info)(supports_get_mem_info());
}

// CudaMemoryResource

Napi::FunctionReference CudaMemoryResource::constructor;

Napi::Object CudaMemoryResource::Init(Napi::Env env, Napi::Object exports) {
  exports.Set("CudaMemoryResource", [&]() {
    CudaMemoryResource::constructor = Napi::Persistent(DefineClass(
      env,
      "CudaMemoryResource",
      {InstanceMethod("isEqual", &CudaMemoryResource::is_equal),
       InstanceMethod("getMemInfo", &CudaMemoryResource::get_mem_info),
       InstanceAccessor(
         "supportsStreams", &CudaMemoryResource::supports_streams, nullptr, napi_enumerable),
       InstanceAccessor("supportsGetMemInfo",
                        &CudaMemoryResource::supports_get_mem_info,
                        nullptr,
                        napi_enumerable)}));
    CudaMemoryResource::constructor.SuppressDestruct();
    return CudaMemoryResource::constructor.Value();
  }());
  return exports;
}

Napi::Object CudaMemoryResource::New() {
  return CudaMemoryResource::New(Device::active_device_id());
}

Napi::Object CudaMemoryResource::New(int32_t device_id) {
  CPPToNapiValues args{CudaMemoryResource::constructor.Env()};
  return CudaMemoryResource::constructor.New(args(device_id));
}

CudaMemoryResource::CudaMemoryResource(CallbackArgs const& args)
  : Napi::ObjectWrap<CudaMemoryResource>(args) {
  if (!args[0].IsNumber()) {
    device_id_ = Device::active_device_id();
    mr_.reset(new rmm::mr::cuda_memory_resource());
  } else {
    device_id_ = args[0];
    mr_.reset(rmm::mr::get_per_device_resource(rmm::cuda_device_id(device_id_)), [](auto* p) {});
  }
}

// ManagedMemoryResource

Napi::FunctionReference ManagedMemoryResource::constructor;

Napi::Object ManagedMemoryResource::Init(Napi::Env env, Napi::Object exports) {
  exports.Set("ManagedMemoryResource", [&]() {
    (ManagedMemoryResource::constructor = Napi::Persistent(DefineClass(
       env,
       "ManagedMemoryResource",
       {InstanceMethod("isEqual", &ManagedMemoryResource::is_equal),
        InstanceMethod("getMemInfo", &ManagedMemoryResource::get_mem_info),
        InstanceAccessor(
          "supportsStreams", &ManagedMemoryResource::supports_streams, nullptr, napi_enumerable),
        InstanceAccessor("supportsGetMemInfo",
                         &ManagedMemoryResource::supports_get_mem_info,
                         nullptr,
                         napi_enumerable)})))
      .SuppressDestruct();
    return ManagedMemoryResource::constructor.Value();
  }());

  return exports;
}

ManagedMemoryResource::ManagedMemoryResource(CallbackArgs const& args)
  : Napi::ObjectWrap<ManagedMemoryResource>(args) {
  device_id_ = Device::active_device_id();
  mr_.reset(new rmm::mr::managed_memory_resource());
}

// PoolMemoryResource

Napi::FunctionReference PoolMemoryResource::constructor;

Napi::Object PoolMemoryResource::Init(Napi::Env env, Napi::Object exports) {
  exports.Set("PoolMemoryResource", [&]() {
    PoolMemoryResource::constructor = Napi::Persistent(DefineClass(
      env,
      "PoolMemoryResource",
      {InstanceMethod("isEqual", &PoolMemoryResource::is_equal),
       InstanceMethod("getMemInfo", &PoolMemoryResource::get_mem_info),
       InstanceAccessor(
         "upstreamMemoryResource", &PoolMemoryResource::get_upstream_mr, nullptr, napi_enumerable),
       InstanceAccessor(
         "supportsStreams", &PoolMemoryResource::supports_streams, nullptr, napi_enumerable),
       InstanceAccessor("supportsGetMemInfo",
                        &PoolMemoryResource::supports_get_mem_info,
                        nullptr,
                        napi_enumerable)}));
    PoolMemoryResource::constructor.SuppressDestruct();
    return PoolMemoryResource::constructor.Value();
  }());
  return exports;
}

PoolMemoryResource::PoolMemoryResource(CallbackArgs const& args)
  : Napi::ObjectWrap<PoolMemoryResource>(args) {
  NODE_CUDA_EXPECT(args.IsConstructCall(), "PoolMemoryResource constructor requires 'new'");

  NODE_CUDA_EXPECT(MemoryResource::is_instance(args[0]),
                   "PoolMemoryResource constructor expects an upstream MemoryResource from which "
                   "to allocate blocks for the pool.");

  size_t const initial_pool_size = args[1].IsNumber() ? args[1] : -1;
  size_t const maximum_pool_size = args[2].IsNumber() ? args[2] : -1;

  mr_.reset(new rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource>(
    args[0],
    initial_pool_size == -1 ? thrust::nullopt : thrust::make_optional(initial_pool_size),
    maximum_pool_size == -1 ? thrust::nullopt : thrust::make_optional(maximum_pool_size)));

  upstream_mr_.Reset(args[0], 1);
}

int32_t PoolMemoryResource::device() const {
  return NapiToCPP(upstream_mr_.Value()).operator rmm::cuda_device_id().value();
}

Napi::Value PoolMemoryResource::get_upstream_mr(Napi::CallbackInfo const& info) {
  return upstream_mr_.Value();
}

// FixedSizeMemoryResource

Napi::FunctionReference FixedSizeMemoryResource::constructor;

Napi::Object FixedSizeMemoryResource::Init(Napi::Env env, Napi::Object exports) {
  exports.Set("FixedSizeMemoryResource", [&]() {
    FixedSizeMemoryResource::constructor = Napi::Persistent(DefineClass(
      env,
      "FixedSizeMemoryResource",
      {InstanceMethod("isEqual", &FixedSizeMemoryResource::is_equal),
       InstanceMethod("getMemInfo", &FixedSizeMemoryResource::get_mem_info),
       InstanceAccessor("upstreamMemoryResource",
                        &FixedSizeMemoryResource::get_upstream_mr,
                        nullptr,
                        napi_enumerable),
       InstanceAccessor(
         "supportsStreams", &FixedSizeMemoryResource::supports_streams, nullptr, napi_enumerable),
       InstanceAccessor("supportsGetMemInfo",
                        &FixedSizeMemoryResource::supports_get_mem_info,
                        nullptr,
                        napi_enumerable)}));
    FixedSizeMemoryResource::constructor.SuppressDestruct();
    return FixedSizeMemoryResource::constructor.Value();
  }());
  return exports;
}

FixedSizeMemoryResource::FixedSizeMemoryResource(CallbackArgs const& args)
  : Napi::ObjectWrap<FixedSizeMemoryResource>(args) {
  NODE_CUDA_EXPECT(args.IsConstructCall(), "FixedSizeMemoryResource constructor requires 'new'");

  NODE_CUDA_EXPECT(
    MemoryResource::is_instance(args[0]),
    "FixedSizeMemoryResource constructor expects an upstream MemoryResource from which "
    "to allocate blocks for the pool.");

  rmm::mr::device_memory_resource* r = args[0];
  size_t const block_size            = args[1].IsNumber() ? args[1] : 1 << 20;
  size_t const blocks_to_preallocate = args[2].IsNumber() ? args[2] : 128;

  mr_.reset(new rmm::mr::fixed_size_memory_resource<rmm::mr::device_memory_resource>(
    r, block_size, blocks_to_preallocate));

  upstream_mr_.Reset(args[0], 1);
}

int32_t FixedSizeMemoryResource::device() const {
  return NapiToCPP(upstream_mr_.Value()).operator rmm::cuda_device_id().value();
}

Napi::Value FixedSizeMemoryResource::get_upstream_mr(Napi::CallbackInfo const& info) {
  return upstream_mr_.Value();
}

// BinningMemoryResource

Napi::FunctionReference BinningMemoryResource::constructor;

Napi::Object BinningMemoryResource::Init(Napi::Env env, Napi::Object exports) {
  exports.Set("BinningMemoryResource", [&]() {
    BinningMemoryResource::constructor = Napi::Persistent(DefineClass(
      env,
      "BinningMemoryResource",
      {InstanceMethod("isEqual", &BinningMemoryResource::is_equal),
       InstanceMethod("getMemInfo", &BinningMemoryResource::get_mem_info),
       InstanceMethod("addBin", &BinningMemoryResource::add_bin),
       InstanceAccessor("upstreamMemoryResource",
                        &BinningMemoryResource::get_upstream_mr,
                        nullptr,
                        napi_enumerable),
       InstanceAccessor(
         "supportsStreams", &BinningMemoryResource::supports_streams, nullptr, napi_enumerable),
       InstanceAccessor("supportsGetMemInfo",
                        &BinningMemoryResource::supports_get_mem_info,
                        nullptr,
                        napi_enumerable)}));
    BinningMemoryResource::constructor.SuppressDestruct();
    return BinningMemoryResource::constructor.Value();
  }());
  return exports;
}

BinningMemoryResource::BinningMemoryResource(CallbackArgs const& args)
  : Napi::ObjectWrap<BinningMemoryResource>(args) {
  NODE_CUDA_EXPECT(args.IsConstructCall(), "BinningMemoryResource constructor requires 'new'");

  NODE_CUDA_EXPECT(MemoryResource::is_instance(args[0]),
                   "BinningMemoryResource constructor expects an upstream MemoryResource to use "
                   "for allocations larger than any of the bins.");

  rmm::mr::device_memory_resource* mr = args[0];
  int8_t const min_size_exponent      = args[1].IsNumber() ? args[1] : -1;
  int8_t const max_size_exponent      = args[2].IsNumber() ? args[2] : -1;

  mr_.reset(min_size_exponent <= -1 || max_size_exponent <= -1
              ? new rmm::mr::binning_memory_resource<rmm::mr::device_memory_resource>(mr)
              : new rmm::mr::binning_memory_resource<rmm::mr::device_memory_resource>(
                  mr, min_size_exponent, max_size_exponent));

  upstream_mr_.Reset(args[0], 1);
}

int32_t BinningMemoryResource::device() const {
  return NapiToCPP(upstream_mr_.Value()).operator rmm::cuda_device_id().value();
}

Napi::Value BinningMemoryResource::get_upstream_mr(Napi::CallbackInfo const& info) {
  return upstream_mr_.Value();
}

void BinningMemoryResource::add_bin(size_t allocation_size) {
  get_bin_mr()->add_bin(allocation_size);
}
void BinningMemoryResource::add_bin(size_t allocation_size, Napi::Object const& bin_resource) {
  bin_mrs_.push_back(Napi::ObjectReference::New(bin_resource, 1));
  get_bin_mr()->add_bin(allocation_size, NapiToCPP(bin_resource));
}

Napi::Value BinningMemoryResource::add_bin(Napi::CallbackInfo const& info) {
  CallbackArgs const args{info};
  switch (info.Length()) {
    case 1: add_bin(args[0].operator size_t()); break;
    case 2: add_bin(args[0].operator size_t(), args[1]); break;
    default:
      NODE_CUDA_EXPECT(
        false, "add_bin expects numeric allocation_size and optional MemoryResource arguments.");
  }
  return info.Env().Undefined();
}

// LoggingResourceAdapter

Napi::FunctionReference LoggingResourceAdapter::constructor;

Napi::Object LoggingResourceAdapter::Init(Napi::Env env, Napi::Object exports) {
  exports.Set("LoggingResourceAdapter", [&]() {
    LoggingResourceAdapter::constructor = Napi::Persistent(DefineClass(
      env,
      "LoggingResourceAdapter",
      {InstanceMethod("isEqual", &LoggingResourceAdapter::is_equal),
       InstanceMethod("getMemInfo", &LoggingResourceAdapter::get_mem_info),
       InstanceAccessor(
         "logFilePath", &LoggingResourceAdapter::get_file_path, nullptr, napi_enumerable),
       InstanceAccessor("upstreamMemoryResource",
                        &LoggingResourceAdapter::get_upstream_mr,
                        nullptr,
                        napi_enumerable),
       InstanceAccessor(
         "supportsStreams", &LoggingResourceAdapter::supports_streams, nullptr, napi_enumerable),
       InstanceAccessor("supportsGetMemInfo",
                        &LoggingResourceAdapter::supports_get_mem_info,
                        nullptr,
                        napi_enumerable)}));
    LoggingResourceAdapter::constructor.SuppressDestruct();
    return LoggingResourceAdapter::constructor.Value();
  }());
  return exports;
}

LoggingResourceAdapter::LoggingResourceAdapter(CallbackArgs const& args)
  : Napi::ObjectWrap<LoggingResourceAdapter>(args) {
  NODE_CUDA_EXPECT(args.IsConstructCall(), "LoggingResourceAdapter constructor requires 'new'");

  NODE_CUDA_EXPECT(MemoryResource::is_instance(args[0]),
                   "LoggingResourceAdapter constructor expects an upstream MemoryResource.");

  log_file_path_  = args[1].operator std::string();
  bool auto_flush = args[2].IsBoolean() ? args[2] : false;

  if (log_file_path_ == "") {
    log_file_path_ = args.Env()
                       .Global()
                       .Get("process")
                       .ToObject()
                       .Get("env")
                       .ToObject()
                       .Get("RMM_LOG_FILE")
                       .ToString();
  }

  NODE_CUDA_EXPECT(log_file_path_ != "",
                   "LoggingResourceAdapter constructor expects an RMM log file name string "
                   "argument or RMM_LOG_FILE environment variable");

  mr_.reset(new rmm::mr::logging_resource_adaptor<rmm::mr::device_memory_resource>(
    args[0], log_file_path_, auto_flush));

  upstream_mr_.Reset(args[0], 1);
}

int32_t LoggingResourceAdapter::device() const {
  return NapiToCPP(upstream_mr_.Value()).operator rmm::cuda_device_id().value();
}

Napi::Value LoggingResourceAdapter::get_upstream_mr(Napi::CallbackInfo const& info) {
  return upstream_mr_.Value();
}

Napi::Value LoggingResourceAdapter::flush(Napi::CallbackInfo const& info) {
  flush();
  return info.Env().Undefined();
}

Napi::Value LoggingResourceAdapter::get_file_path(Napi::CallbackInfo const& info) {
  return CPPToNapi(info)(get_file_path());
}

}  // namespace nv
