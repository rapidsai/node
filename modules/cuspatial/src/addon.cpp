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

#include <node_cuspatial/addon.hpp>
#include <node_cuspatial/geometry.hpp>
#include <node_cuspatial/quadtree.hpp>

#include <nv_node/macros.hpp>

#include <napi.h>

namespace nv {
Napi::Value cuspatialInit(Napi::CallbackInfo const& info) {
  // todo
  return info.This();
}
}  // namespace nv

Napi::Object initModule(Napi::Env env, Napi::Object exports) {
  EXPORT_FUNC(env, exports, "init", nv::cuspatialInit);
  EXPORT_FUNC(env, exports, "createQuadtree", nv::create_quadtree);
  EXPORT_FUNC(env,
              exports,
              "findQuadtreeAndBoundingBoxIntersections",
              nv::quadtree_bounding_box_intersections);
  EXPORT_FUNC(env, exports, "findPointsInPolygons", nv::find_points_in_polygons);
  EXPORT_FUNC(
    env, exports, "findPolylineNearestToEachPoint", nv::find_polyline_nearest_to_each_point);
  EXPORT_FUNC(env, exports, "computePolygonBoundingBoxes", nv::compute_polygon_bounding_boxes);
  EXPORT_FUNC(env, exports, "computePolylineBoundingBoxes", nv::compute_polyline_bounding_boxes);
  return exports;
}

NODE_API_MODULE(node_cuspatial, initModule);
