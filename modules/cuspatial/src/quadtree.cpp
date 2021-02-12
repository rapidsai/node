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
#include <nv_node/utilities/wrap.hpp>

namespace nv {

Napi::Value create_quadtree(CallbackArgs const& args) {
  auto xs                             = Column::Unwrap(args[0]);
  auto ys                             = Column::Unwrap(args[1]);
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
    } catch (cuspatial::logic_error const& err) { throw Napi::Error::New(args.Env(), err.what()); }
  }();
  auto output = Napi::Object::New(args.Env());
  auto names  = Napi::Array::New(args.Env(), 5);
  names.Set(0u, "key");
  names.Set(1u, "level");
  names.Set(2u, "is_quad");
  names.Set(3u, "length");
  names.Set(4u, "offset");
  output.Set("names", names);
  output.Set("table", Table::New(std::move(result.second)));
  output.Set("keyMap", Column::New(std::move(result.first))->Value());
  return output;
}

Napi::Value quadtree_bounding_box_intersections(CallbackArgs const& args) {
  auto quadtree                       = Table::Unwrap(args[0]);
  auto poly_bbox                      = Table::Unwrap(args[1]);
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
    } catch (cuspatial::logic_error const& err) { throw Napi::Error::New(args.Env(), err.what()); }
  }();
  auto output = Napi::Object::New(args.Env());
  auto names  = Napi::Array::New(args.Env(), 2);
  names.Set(0u, "polygon_index");
  names.Set(1u, "point_index");
  output.Set("names", names);
  output.Set("table", Table::New(std::move(result)));
  return output;
}

Napi::Value find_points_in_polygons(CallbackArgs const& args) {
  auto intersections                  = Table::Unwrap(args[0]);
  auto quadtree                       = Table::Unwrap(args[1]);
  auto point_indices                  = Column::Unwrap(args[2]);
  auto x                              = Column::Unwrap(args[3]);
  auto y                              = Column::Unwrap(args[4]);
  auto polygon_offsets                = Column::Unwrap(args[5]);
  auto ring_offsets                   = Column::Unwrap(args[6]);
  auto polygon_points_x               = Column::Unwrap(args[7]);
  auto polygon_points_y               = Column::Unwrap(args[8]);
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
    } catch (cuspatial::logic_error const& err) { throw Napi::Error::New(args.Env(), err.what()); }
  }();
  auto output = Napi::Object::New(args.Env());
  auto names  = Napi::Array::New(args.Env(), 2);
  names.Set(0u, "polygon_index");
  names.Set(1u, "point_index");
  output.Set("names", names);
  output.Set("table", Table::New(std::move(result)));
  return output;
}

Napi::Value find_polyline_nearest_to_each_point(CallbackArgs const& args) {
  auto intersections                  = Table::Unwrap(args[0]);
  auto quadtree                       = Table::Unwrap(args[1]);
  auto point_indices                  = Column::Unwrap(args[2]);
  auto x                              = Column::Unwrap(args[3]);
  auto y                              = Column::Unwrap(args[4]);
  auto polyline_offsets               = Column::Unwrap(args[5]);
  auto polyline_points_x              = Column::Unwrap(args[6]);
  auto polyline_points_y              = Column::Unwrap(args[7]);
  rmm::mr::device_memory_resource* mr = args[8];
  auto result                         = [&]() {
    try {
      return cuspatial::quadtree_point_to_nearest_polyline(*intersections,
                                                           *quadtree,
                                                           *point_indices,
                                                           *x,
                                                           *y,
                                                           *polyline_offsets,
                                                           *polyline_points_x,
                                                           *polyline_points_y,
                                                           mr);
    } catch (cuspatial::logic_error const& err) { throw Napi::Error::New(args.Env(), err.what()); }
  }();
  auto output = Napi::Object::New(args.Env());
  auto names  = Napi::Array::New(args.Env(), 3);
  names.Set(0u, "point_index");
  names.Set(1u, "polyline_index");
  names.Set(2u, "distance");
  output.Set("names", names);
  output.Set("table", Table::New(std::move(result)));
  return output;
}

}  // namespace nv
