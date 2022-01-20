// Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

#include <cudf/io/datasource.hpp>
#include <node_cudf/table.hpp>
#include <nv_node/utilities/args.hpp>

#include <las.hpp>

#include <fstream>
#include <iterator>
#include <vector>

struct rapidsai_io : public nv::EnvLocalAddon, public Napi::Addon<rapidsai_io> {
  rapidsai_io(Napi::Env env, Napi::Object exports) : nv::EnvLocalAddon(env, exports) {
    DefineAddon(exports,
                {InstanceMethod("init", &rapidsai_io::InitAddon),
                 InstanceValue("_cpp_exports", _cpp_exports.Value()),
                 InstanceMethod<&rapidsai_io::read_las>("readLasTable")});
  }

 private:
  Napi::Value read_las(Napi::CallbackInfo const& info) {
    nv::CallbackArgs args{info};
    auto env = info.Env();

    std::string path                    = args[0];
    rmm::mr::device_memory_resource* mr = args[1];

    auto [names, table] = nv::read_las(cudf::io::datasource::create(path), mr);

    auto result = Napi::Object::New(env);

    auto table_names = Napi::Array::New(env, names.size());
    for (size_t i = 0; i < names.size(); ++i) {
      table_names.Set(i, Napi::String::New(env, names[i]));
    }

    result.Set("names", table_names);
    result.Set("table", nv::Table::New(env, std::move(table)));

    return result;
  }
};

NODE_API_ADDON(rapidsai_io);
