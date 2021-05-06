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

#include "node_cuspatial/geometry.hpp"
#include "node_cuspatial/quadtree.hpp"

#include <nv_node/addon.hpp>

struct node_cuspatial : public nv::EnvLocalAddon, public Napi::Addon<node_cuspatial> {
  node_cuspatial(Napi::Env const& env, Napi::Object exports) : nv::EnvLocalAddon(env, exports) {
    DefineAddon(exports,
                {
                  InstanceMethod("init", &node_cuspatial::InitAddon),
                  InstanceValue("_cpp_exports", _cpp_exports.Value()),
                  InstanceMethod<&node_cuspatial::create_quadtree>("createQuadtree"),
                  InstanceMethod<&node_cuspatial::quadtree_bounding_box_intersections>(
                    "findQuadtreeAndBoundingBoxIntersections"),
                  InstanceMethod<&node_cuspatial::find_points_in_polygons>("findPointsInPolygons"),
                  InstanceMethod<&node_cuspatial::find_polyline_nearest_to_each_point>(
                    "findPolylineNearestToEachPoint"),
                  InstanceMethod<&node_cuspatial::compute_polygon_bounding_boxes>(
                    "computePolygonBoundingBoxes"),
                  InstanceMethod<&node_cuspatial::compute_polyline_bounding_boxes>(
                    "computePolylineBoundingBoxes"),
                });
  }

 private:
  Napi::Value create_quadtree(Napi::CallbackInfo const& info) { return nv::create_quadtree(info); }
  Napi::Value quadtree_bounding_box_intersections(Napi::CallbackInfo const& info) {
    return nv::quadtree_bounding_box_intersections(info);
  }
  Napi::Value find_points_in_polygons(Napi::CallbackInfo const& info) {
    return nv::find_points_in_polygons(info);
  }
  Napi::Value find_polyline_nearest_to_each_point(Napi::CallbackInfo const& info) {
    return nv::find_polyline_nearest_to_each_point(info);
  }
  Napi::Value compute_polygon_bounding_boxes(Napi::CallbackInfo const& info) {
    return nv::compute_polygon_bounding_boxes(info);
  }
  Napi::Value compute_polyline_bounding_boxes(Napi::CallbackInfo const& info) {
    return nv::compute_polyline_bounding_boxes(info);
  }
};

NODE_API_ADDON(node_cuspatial);
