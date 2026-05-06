// Copyright (c) 2021-2026, NVIDIA CORPORATION.
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

#include <arrow/c/bridge.h>
#include <arrow/c/helpers.h>
#include <arrow/gpu/cuda_api.h>
#include <arrow/io/memory.h>
#include <arrow/ipc/api.h>
#include <arrow/ipc/writer.h>
#include <arrow/result.h>
#include <arrow/table.h>

#include <cudf/concatenate.hpp>
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
  auto stream = nv::get_default_stream();

  auto env    = info.Env();
  auto buffer = [&]() {
    stream.synchronize();

    auto c_schema = cudf::to_arrow_schema(this->operator cudf::table_view(),
                                          gather_metadata(info[0].As<Napi::Array>()));
    auto c_batch  = cudf::to_arrow_host(
      this->operator cudf::table_view(), stream, cudf::get_current_device_resource_ref());

    auto schema = arrow::ImportSchema(c_schema.get()).ValueOrDie();
    auto batch  = arrow::ImportRecordBatch(&c_batch->array, schema).ValueOrDie();

    auto sink                = arrow::io::BufferOutputStream::Create().ValueOrDie();
    auto options             = arrow::ipc::IpcWriteOptions::Defaults();
    options.metadata_version = arrow::ipc::MetadataVersion::V4;
    auto writer = arrow::ipc::MakeStreamWriter(sink.get(), schema, options).ValueOrDie();

    {
      auto status = writer->WriteRecordBatch(*batch);
      if (!status.ok()) { NAPI_THROW(Napi::Error::New(env, status.message())); }
    }

    {
      auto status = writer->Close();
      if (!status.ok()) { NAPI_THROW(Napi::Error::New(env, status.message())); }
    }

    stream.synchronize();

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

  auto stream = nv::get_default_stream();

  auto stream_reader =
    RecordBatchStreamReader::Open([&] {
      if (Memory::IsInstance(source) || DeviceBuffer::IsInstance(source)) {
        auto device_manager = CudaDeviceManager::Instance().ValueOrDie();
        auto device_context = device_manager->GetContext(Device::active_device_id()).ValueOrDie();
        buffer_reader.reset(new CudaBufferReader(std::make_shared<CudaBuffer>(
          span.data(), span.size(), device_context, false, IpcMemory::IsInstance(source))));
        return CudaMessageReader::Open(static_cast<CudaBufferReader*>(buffer_reader.get()),
                                       nullptr);
      }
      // If the memory was not allocated via CUDA, assume host
      buffer_reader.reset(
        new BufferReader(std::make_shared<arrow::Buffer>(span.data(), span.size())));
      return MessageReader::Open(buffer_reader.get());
    }())
      .ValueOrDie();

  try {
    auto output = Napi::Object::New(env);

    auto schema = stream_reader->schema();

    output.Set("fields", Napi::Value::From(env, schema->fields()));

    cudf::unique_schema_t c_schema(new ArrowSchema, [](ArrowSchema* schema) {
      ArrowSchemaRelease(schema);
      delete schema;
    });

    {
      auto status = arrow::ExportSchema(*schema, c_schema.get());
      if (!status.ok()) { NAPI_THROW(Napi::Error::New(env, status.message())); }
    }

    auto batches = stream_reader->ToRecordBatches().ValueOrDie();

    std::vector<cudf::unique_device_array_t> c_batches;
    std::vector<cudf::table_view> tables;
    c_batches.reserve(batches.size());
    tables.reserve(batches.size());

    std::transform(
      batches.begin(), batches.end(), std::back_inserter(c_batches), [&](auto const& batch) {
        cudf::unique_device_array_t c_batch(new ArrowDeviceArray, [](ArrowDeviceArray* batch) {
          ArrowDeviceArrayRelease(batch);
          delete batch;
        });
        auto status = arrow::ExportDeviceRecordBatch(*batch, nullptr, c_batch.get());
        if (!status.ok()) { NAPI_THROW(Napi::Error::New(env, status.message())); }
        return c_batch;
      });

    std::transform(
      c_batches.begin(), c_batches.end(), std::back_inserter(tables), [&](auto const& c_batch) {
        return *cudf::from_arrow_device(c_schema.get(), c_batch.get(), stream);
      });

    auto table = [=]() {
      switch (tables.size()) {
        case 0: {
          return std::make_unique<cudf::table>();
        }
        case 1: {
          return std::make_unique<cudf::table>(tables[0], stream);
        }
        default: {
          return cudf::concatenate(tables, stream);
        }
      }
    }();

    stream.synchronize();

    output.Set("table", Table::New(env, std::move(table)));

    return output;
  } catch (std::exception const& e) { NAPI_THROW(Napi::Error::New(env, e.what())); }
}

}  // namespace nv
