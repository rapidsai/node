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

import * as AggregationLayers from '@deck.gl/aggregation-layers';
import {
  CartoBQTilerLayer,
  CartoLayer,
  CartoSQLLayer,
  colorBins,
  colorCategories,
  colorContinuous,
  MAP_TYPES as CARTO_MAP_TYPES
} from '@deck.gl/carto';
import {FirstPersonView, MapView, OrbitView, OrthographicView} from '@deck.gl/core';
import {COORDINATE_SYSTEM} from '@deck.gl/core';
import * as GeoLayers from '@deck.gl/geo-layers';
import {JSONConfiguration, JSONConverter} from '@deck.gl/json';
import * as Layers from '@deck.gl/layers';
import * as MeshLayers from '@deck.gl/mesh-layers';
import {CesiumIonLoader, Tiles3DLoader} from '@loaders.gl/3d-tiles';
import {registerLoaders} from '@loaders.gl/core';
import {CSVLoader} from '@loaders.gl/csv';
import {DracoWorkerLoader} from '@loaders.gl/draco';
import GL from '@luma.gl/constants';

// Note: deck already registers JSONLoader...
registerLoaders([CSVLoader, DracoWorkerLoader]);

const configuration = {
  // Classes that should be instantiatable by JSON converter
  classes: Object.assign(
    // Support `@deck.gl/core` Views
    {MapView, FirstPersonView, OrbitView, OrthographicView},
    // a map of all layers that should be exposes as JSONLayers
    Layers,
    AggregationLayers,
    GeoLayers,
    MeshLayers,
    {CartoLayer, CartoBQTilerLayer, CartoSQLLayer},
    // Any non-standard views
    {}),

  // Functions that should be executed by JSON converter
  functions: {colorBins, colorCategories, colorContinuous},

  // Enumerations that should be available to JSON parser
  // Will be resolved as `<enum-name>.<enum-value>`
  enumerations: {COORDINATE_SYSTEM, GL, CARTO_MAP_TYPES},

  // Constants that should be resolved with the provided values by JSON converter
  constants: {Tiles3DLoader, CesiumIonLoader}
};

export {default as templates} from '@rapidsai/demo-deck-playground/json-examples';

export const converter = new JSONConverter({configuration: new JSONConfiguration(configuration)});
