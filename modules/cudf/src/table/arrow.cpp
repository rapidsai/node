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

#include <node_cudf/column.hpp>
#include <node_cudf/table.hpp>
#include <node_cudf/utilities/dtypes.hpp>

#include <arrow/gpu/cuda_api.h>
#include <arrow/io/memory.h>
#include <arrow/ipc/api.h>
#include <arrow/ipc/writer.h>
#include <arrow/result.h>

#include <cudf/interop.hpp>

#include <memory>

namespace nv {

namespace {

class CudaMessageReader : arrow::ipc::MessageReader {
 public:
  CudaMessageReader(arrow::cuda::CudaBufferReader* stream, arrow::io::BufferReader* schema)
    : stream_(stream), host_schema_reader_(schema){};

  static std::unique_ptr<arrow::ipc::MessageReader> Open(arrow::cuda::CudaBufferReader* stream,
                                                         arrow::io::BufferReader* schema) {
    return std::unique_ptr<arrow::ipc::MessageReader>(new CudaMessageReader(stream, schema));
  }

  arrow::Result<std::unique_ptr<arrow::ipc::Message>> ReadNextMessage() override {
    if (host_schema_reader_ != nullptr) {
      auto message        = arrow::ipc::ReadMessage(host_schema_reader_);
      host_schema_reader_ = nullptr;
      if (message.ok() && *message != nullptr) { return message; }
    }
    return arrow::ipc::ReadMessage(stream_, arrow::default_memory_pool());
  }

 private:
  arrow::cuda::CudaBufferReader* stream_;
  arrow::io::BufferReader* host_schema_reader_ = nullptr;
  std::shared_ptr<arrow::cuda::CudaBufferReader> owned_stream_;
};

std::vector<cudf::column_metadata> gather_metadata(Napi::Array const& names) {
  std::vector<cudf::column_metadata> metadata;
  metadata.reserve(names.Length());
  for (uint32_t i = 0; i < names.Length(); ++i) {
    auto pair = names.Get(i).As<Napi::Array>();
    metadata.push_back({pair.Get(0u).ToString()});
    if (pair.Length() == 2 && pair.Get(1u).IsArray()) {
      metadata[i].children_meta = gather_metadata(pair.Get(1u).As<Napi::Array>());
    }
  }
  return metadata;
}

}  // namespace

Napi::Value Table::to_arrow(Napi::CallbackInfo const& info) {
  auto env    = info.Env();
  auto buffer = [&]() {
    auto table        = cudf::to_arrow(*this, gather_metadata(info[0].As<Napi::Array>()));
    auto sink         = arrow::io::BufferOutputStream::Create().ValueOrDie();
    auto writer       = arrow::ipc::MakeStreamWriter(sink.get(), table->schema()).ValueOrDie();
    auto write_status = writer->WriteTable(*table);
    if (!write_status.ok()) { NAPI_THROW(Napi::Error::New(env, write_status.message())); }
    auto close_status = writer->Close();
    if (!close_status.ok()) { NAPI_THROW(Napi::Error::New(env, close_status.message())); }
    auto buffer = sink->Finish().ValueOrDie();
    auto arybuf = Napi::ArrayBuffer::New(env, buffer->size());
    memcpy(arybuf.Data(), buffer->data(), buffer->size());
    return arybuf;
  }();

  return Napi::Uint8Array::New(env, buffer.ByteLength(), buffer, 0);
}

Napi::Value Table::from_arrow(Napi::CallbackInfo const& info) {
  using namespace arrow::cuda;
  using namespace arrow::io;
  using namespace arrow::ipc;

  CallbackArgs args{info};
  auto env = info.Env();

  auto source        = args[0];
  Span<uint8_t> span = args[0];
  std::unique_ptr<InputStream> buffer_reader;

  auto stream_reader =
    RecordBatchStreamReader::Open([&] {
      if (Memory::IsInstance(source) || DeviceBuffer::IsInstance(source)) {
        auto device_manager = CudaDeviceManager::Instance().ValueOrDie();
        auto device_context = device_manager->GetContext(Device::active_device_id()).ValueOrDie();
        buffer_reader.reset(new CudaBufferReader(std::make_shared<CudaBuffer>(
          span.data(), span.size(), device_context, false, IpcMemory::IsInstance(source))));
        return nv::CudaMessageReader::Open(static_cast<CudaBufferReader*>(buffer_reader.get()),
                                           nullptr);
      }
      // If the memory was not allocated via CUDA, assume host
      buffer_reader.reset(
        new BufferReader(std::make_shared<arrow::Buffer>(span.data(), span.size())));
      return MessageReader::Open(buffer_reader.get());
    }())
      .ValueOrDie();

  try {
    auto arrow_table = stream_reader->ToTable().ValueOrDie();

    auto output = Napi::Object::New(env);
    output.Set("table", Table::New(env, cudf::from_arrow(*arrow_table)));
    output.Set("fields", Napi::Value::From(env, stream_reader->schema()->fields()));

    return output;
  } catch (std::exception const& e) { NAPI_THROW(Napi::Error::New(env, e.what())); }
}

}  // namespace nv
