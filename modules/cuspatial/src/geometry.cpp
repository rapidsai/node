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

#include <node_cuspatial/geometry.hpp>

#include <node_cudf/column.hpp>
#include <node_cudf/table.hpp>

#include <node_rmm/utilities/napi_to_cpp.hpp>

#include <cuspatial/coordinate_transform.hpp>
#include <cuspatial/error.hpp>
#include <cuspatial/linestring_bounding_box.hpp>
#include <cuspatial/polygon_bounding_box.hpp>

#include <nv_node/utilities/args.hpp>

namespace nv {

Napi::Value compute_polygon_bounding_boxes(CallbackArgs const& args) {
  Column::wrapper_t poly_offsets      = args[0];
  Column::wrapper_t ring_offsets      = args[1];
  Column::wrapper_t point_x           = args[2];
  Column::wrapper_t point_y           = args[3];
  rmm::mr::device_memory_resource* mr = args[4];
  auto result                         = [&]() {
    try {
      return cuspatial::polygon_bounding_boxes(
        *poly_offsets, *ring_offsets, *point_x, *point_y, 0.0, mr);
    } catch (std::exception const& e) { throw Napi::Error::New(args.Env(), e.what()); }
  }();
  auto output = Napi::Object::New(args.Env());
  auto names  = Napi::Array::New(args.Env(), 4);
  names.Set(0u, "x_min");
  names.Set(1u, "y_min");
  names.Set(2u, "x_max");
  names.Set(3u, "y_max");
  output.Set("names", names);
  output.Set("table", Table::New(args.Env(), std::move(result)));
  return output;
}

Napi::Value compute_polyline_bounding_boxes(CallbackArgs const& args) {
  Column::wrapper_t poly_offsets      = args[0];
  Column::wrapper_t point_x           = args[1];
  Column::wrapper_t point_y           = args[2];
  double expansion_radius             = args[3];
  rmm::mr::device_memory_resource* mr = args[4];
  auto result                         = [&]() {
    try {
      return cuspatial::linestring_bounding_boxes(
        *poly_offsets, *point_x, *point_y, expansion_radius, mr);
    } catch (std::exception const& e) { throw Napi::Error::New(args.Env(), e.what()); }
  }();
  auto output = Napi::Object::New(args.Env());
  auto names  = Napi::Array::New(args.Env(), 4);
  names.Set(0u, "x_min");
  names.Set(1u, "y_min");
  names.Set(2u, "x_max");
  names.Set(3u, "y_max");
  output.Set("names", names);
  output.Set("table", Table::New(args.Env(), std::move(result)));
  return output;
}

Napi::Value lonlat_to_cartesian(CallbackArgs const& args) {
  double origin_lon                   = args[0];
  double origin_lat                   = args[1];
  Column::wrapper_t lons_column       = args[2];
  Column::wrapper_t lats_column       = args[3];
  rmm::mr::device_memory_resource* mr = args[4];
  auto result                         = [&]() {
    try {
      return cuspatial::lonlat_to_cartesian(origin_lon, origin_lat, *lons_column, *lats_column, mr);
    } catch (std::exception const& e) { throw Napi::Error::New(args.Env(), e.what()); }
  }();
  auto output = Napi::Object::New(args.Env());
  output.Set("x", Column::New(args.Env(), std::move(result.first)));
  output.Set("y", Column::New(args.Env(), std::move(result.second)));
  return output;
}

}  // namespace nv
