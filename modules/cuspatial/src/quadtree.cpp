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

#include <node_rmm/utilities/napi_to_cpp.hpp>

#include <cuspatial/error.hpp>
#include <cuspatial/point_quadtree.hpp>
#include <cuspatial/spatial_join.hpp>

#include <nv_node/utilities/args.hpp>

namespace nv {

Napi::Value create_quadtree(CallbackArgs const& args) {
  Column::wrapper_t xs                = args[0];
  Column::wrapper_t ys                = args[1];
  double x_min                        = args[2];
  double x_max                        = args[3];
  double y_min                        = args[4];
  double y_max                        = args[5];
  double scale                        = args[6];
  int8_t max_depth                    = args[7];
  cudf::size_type min_size            = args[8];
  rmm::mr::device_memory_resource* mr = args[9];
  auto result                         = [&]() {
    try {
      return cuspatial::quadtree_on_points(
        *xs, *ys, x_min, x_max, y_min, y_max, scale, max_depth, min_size, mr);
    } catch (std::exception const& e) { throw Napi::Error::New(args.Env(), e.what()); }
  }();
  auto output = Napi::Object::New(args.Env());
  auto names  = Napi::Array::New(args.Env(), 5);
  names.Set(0u, "key");
  names.Set(1u, "level");
  names.Set(2u, "is_quad");
  names.Set(3u, "length");
  names.Set(4u, "offset");
  output.Set("names", names);
  output.Set("table", Table::New(args.Env(), std::move(result.second)));
  output.Set("keyMap", Column::New(args.Env(), std::move(result.first)));
  return output;
}

Napi::Value quadtree_bounding_box_intersections(CallbackArgs const& args) {
  Table::wrapper_t quadtree           = args[0];
  Table::wrapper_t poly_bbox          = args[1];
  double x_min                        = args[2];
  double x_max                        = args[3];
  double y_min                        = args[4];
  double y_max                        = args[5];
  double scale                        = args[6];
  int8_t max_depth                    = args[7];
  rmm::mr::device_memory_resource* mr = args[8];
  auto result                         = [&]() {
    try {
      return cuspatial::join_quadtree_and_bounding_boxes(
        *quadtree, *poly_bbox, x_min, x_max, y_min, y_max, scale, max_depth, mr);
    } catch (std::exception const& e) { throw Napi::Error::New(args.Env(), e.what()); }
  }();
  auto output = Napi::Object::New(args.Env());
  auto names  = Napi::Array::New(args.Env(), 2);
  names.Set(0u, "polygon_index");
  names.Set(1u, "point_index");
  output.Set("names", names);
  output.Set("table", Table::New(args.Env(), std::move(result)));
  return output;
}

Napi::Value find_points_in_polygons(CallbackArgs const& args) {
  Table::wrapper_t intersections      = args[0];
  Table::wrapper_t quadtree           = args[1];
  Column::wrapper_t point_indices     = args[2];
  Column::wrapper_t x                 = args[3];
  Column::wrapper_t y                 = args[4];
  Column::wrapper_t polygon_offsets   = args[5];
  Column::wrapper_t ring_offsets      = args[6];
  Column::wrapper_t polygon_points_x  = args[7];
  Column::wrapper_t polygon_points_y  = args[8];
  rmm::mr::device_memory_resource* mr = args[9];
  auto result                         = [&]() {
    try {
      return cuspatial::quadtree_point_in_polygon(*intersections,
                                                  *quadtree,
                                                  *point_indices,
                                                  *x,
                                                  *y,
                                                  *polygon_offsets,
                                                  *ring_offsets,
                                                  *polygon_points_x,
                                                  *polygon_points_y,
                                                  mr);
    } catch (std::exception const& e) { throw Napi::Error::New(args.Env(), e.what()); }
  }();
  auto output = Napi::Object::New(args.Env());
  auto names  = Napi::Array::New(args.Env(), 2);
  names.Set(0u, "polygon_index");
  names.Set(1u, "point_index");
  output.Set("names", names);
  output.Set("table", Table::New(args.Env(), std::move(result)));
  return output;
}

Napi::Value find_polyline_nearest_to_each_point(CallbackArgs const& args) {
  Table::wrapper_t intersections      = args[0];
  Table::wrapper_t quadtree           = args[1];
  Column::wrapper_t point_indices     = args[2];
  Column::wrapper_t x                 = args[3];
  Column::wrapper_t y                 = args[4];
  Column::wrapper_t polyline_offsets  = args[5];
  Column::wrapper_t polyline_points_x = args[6];
  Column::wrapper_t polyline_points_y = args[7];
  rmm::mr::device_memory_resource* mr = args[8];
  auto result                         = [&]() {
    try {
      return cuspatial::quadtree_point_to_nearest_linestring(*intersections,
                                                             *quadtree,
                                                             *point_indices,
                                                             *x,
                                                             *y,
                                                             *polyline_offsets,
                                                             *polyline_points_x,
                                                             *polyline_points_y,
                                                             mr);
    } catch (std::exception const& e) { throw Napi::Error::New(args.Env(), e.what()); }
  }();
  auto output = Napi::Object::New(args.Env());
  auto names  = Napi::Array::New(args.Env(), 3);
  names.Set(0u, "point_index");
  names.Set(1u, "polyline_index");
  names.Set(2u, "distance");
  output.Set("names", names);
  output.Set("table", Table::New(args.Env(), std::move(result)));
  return output;
}

}  // namespace nv
