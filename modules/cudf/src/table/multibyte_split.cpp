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

#include <cudf/io/text/data_chunk_source_factories.hpp>
#include <cudf/io/text/multibyte_split.hpp>
#include <node_cudf/utilities/metadata.hpp>

namespace nv {

namespace {

Napi::Value read_text_files(Napi::CallbackInfo const& info,
                            std::string const& filename,
                            std::string const& delimiter) {
  auto env        = info.Env();
  auto datasource = cudf::io::text::make_source_from_file(filename);
  return Column::New(env, cudf::io::text::multibyte_split(*datasource, delimiter));
}

}  // namespace

Napi::Value Column::read_text(Napi::CallbackInfo const& info) {
  CallbackArgs args{info};

  if (args.Length() != 2) {
    NAPI_THROW(
      Napi::Error::New(info.Env(), "read_text expects a filename and an optional delimiter"));
  }

  auto source    = args[0];
  auto delimiter = args[1];

  try {
    return ::nv::read_text_files(info, source, delimiter);
  } catch (cudf::logic_error const& err) { NAPI_THROW(Napi::Error::New(info.Env(), err.what())); }
}

}  // namespace nv
