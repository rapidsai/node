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

#include <node_cuspatial/quadtree.hpp>

#include <node_cudf/column.hpp>
#include <node_cudf/table.hpp>

#include <cuspatial/point_quadtree.hpp>

#include <nv_node/utilities/args.hpp>
#include <nv_node/utilities/wrap.hpp>

namespace nv {
Napi::Value create_quadtree(CallbackArgs const& args) {
  auto xs = Column::Unwrap(args[0]);
  auto ys = Column::Unwrap(args[1]);
  double x_min{args[2]};
  double x_max{args[3]};
  double y_min{args[4]};
  double y_max{args[5]};
  double scale{args[6]};
  int8_t max_depth{args[7]};
  cudf::size_type min_size{args[8]};
  auto result =
    cuspatial::quadtree_on_points(*xs, *ys, x_min, x_max, y_min, y_max, scale, max_depth, min_size);
  auto output = Napi::Object::New(args.Env());
  auto names  = Napi::Array::New(args.Env(), 5);
  names.Set(0u, "key");
  names.Set(1u, "level");
  names.Set(2u, "isQuad");
  names.Set(3u, "length");
  names.Set(4u, "offset");
  output.Set("names", names);
  output.Set("table", Table::New(std::move(result.second)));
  output.Set("keyMap", Column::New(std::move(result.first))->Value());
  return output;
}
}  // namespace nv
