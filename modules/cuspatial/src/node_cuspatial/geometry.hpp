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
 * @brief Compute the minimum bounding-boxes for a set of polygons.
 *
 * @param args CallbackArgs JavaScript arguments list.
 */
Napi::Value compute_polygon_bounding_boxes(CallbackArgs const& args);

/**
 * @brief Compute the minimum bounding-boxes for a set of polylines.
 *
 * @param args CallbackArgs JavaScript arguments list.
 */
Napi::Value compute_polyline_bounding_boxes(CallbackArgs const& args);

/**
 * @brief Convert lon/lat coordinate columns into cartesian coordinates.
 *
 * @param args CallbackArgs JavaScript arguments list.
 */
Napi::Value lonlat_to_cartesian(CallbackArgs const& args);

}  // namespace nv
