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

#include <memory>
#include <node_cudf/column.hpp>
#include <node_cudf/table.hpp>

#include <arrow/io/memory.h>
#include <arrow/ipc/writer.h>

#include <cudf/interop.hpp>
#include <cudf/ipc.hpp>

namespace nv {

namespace {

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
    auto writer       = arrow::ipc::NewStreamWriter(sink.get(), table->schema()).ValueOrDie();
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

  auto device_manager = CudaDeviceManager::Instance().ValueOrDie();
  auto context        = device_manager->GetContext(Device::active_device_id()).ValueOrDie();

  Span<uint8_t> span = args[0];
  auto buffer_reader = std::unique_ptr<InputStream>(nullptr);

  auto message_reader = [&] {
    uint32_t memory_type  = CU_MEMORYTYPE_HOST;
    CUresult const status = cuPointerGetAttribute(
      &memory_type, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, reinterpret_cast<CUdeviceptr>(span.data()));

    // If the memory was not allocated via Cuda at all, this error is returned -- assume host
    if (status == CUDA_ERROR_INVALID_VALUE or memory_type == CU_MEMORYTYPE_HOST) {
      buffer_reader.reset(
        new BufferReader(std::make_shared<arrow::Buffer>(span.data(), span.size())));
      return MessageReader::Open(buffer_reader.get());
    }
    buffer_reader.reset(new CudaBufferReader(std::make_shared<CudaBuffer>(
      span.data(), span.size(), context, false, IpcMemory::IsInstance(args[0]))));
    return CudaMessageReader::Open(static_cast<CudaBufferReader*>(buffer_reader.get()), nullptr);
  }();

  auto stream_reader = RecordBatchStreamReader::Open(std::move(message_reader)).ValueOrDie();

  std::shared_ptr<arrow::Table> arrow_table{};
  auto status = stream_reader->ReadAll(&arrow_table);

  if (!status.ok()) { NAPI_THROW(Napi::Error::New(env, status.message())); }

  auto output = Napi::Object::New(env);

  auto fields = stream_reader->schema()->fields();
  auto names  = Napi::Array::New(env, fields.size());
  for (std::size_t i = 0; i < fields.size(); ++i) { names.Set(i, fields[i]->name()); }
  output.Set("names", names);

  try {
    auto table = cudf::from_arrow(*arrow_table);
    output.Set("table", Table::New(env, std::move(table)));
  } catch (std::exception const& e) { NAPI_THROW(Napi::Error::New(env, e.what())); }

  return output;
}

}  // namespace nv
