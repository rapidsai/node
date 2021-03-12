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

#include <arrow/io/memory.h>
#include <arrow/ipc/writer.h>

#include <cudf/interop.hpp>

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
  auto buffer = [&]() {
    auto table        = cudf::to_arrow(*this, gather_metadata(info[0].As<Napi::Array>()));
    auto sink         = arrow::io::BufferOutputStream::Create().ValueOrDie();
    auto writer       = arrow::ipc::NewStreamWriter(sink.get(), table->schema()).ValueOrDie();
    auto write_status = writer->WriteTable(*table);
    if (!write_status.ok()) { NAPI_THROW(Napi::Error::New(info.Env(), write_status.message())); }
    auto close_status = writer->Close();
    if (!close_status.ok()) { NAPI_THROW(Napi::Error::New(info.Env(), close_status.message())); }
    auto buffer = sink->Finish().ValueOrDie();
    auto arybuf = Napi::ArrayBuffer::New(info.Env(), buffer->size());
    memcpy(arybuf.Data(), buffer->data(), buffer->size());
    return arybuf;
  }();

  return Napi::Uint8Array::New(info.Env(), buffer.ByteLength(), buffer, 0);
}
}  // namespace nv
