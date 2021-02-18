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

#pragma once

#include <nv_node/utilities/args.hpp>

#include <napi.h>

namespace nv {

/**
 * @brief Construct a quadtree from a set of points for a given area-of-interest bounding box.
 *
 * @param args CallbackArgs JavaScript arguments list.
 */
Napi::Value create_quadtree(CallbackArgs const& args);

/**
 * @brief Search a quadtree for polygon or polyline bounding box intersections.
 *
 * @param args CallbackArgs JavaScript arguments list.
 */
Napi::Value quadtree_bounding_box_intersections(CallbackArgs const& args);

/**
 * @brief Test whether the specified points are inside any of the specified polygons.
 */
Napi::Value find_points_in_polygons(CallbackArgs const& args);

/**
 * @brief Finds the nearest polyline to each point in a quadrant, and computes the distances between
 * each point and polyline.
 */
Napi::Value find_polyline_nearest_to_each_point(CallbackArgs const& args);

}  // namespace nv
