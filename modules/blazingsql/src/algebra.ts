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

const java = require('java');

import * as Fs from 'fs';
import * as Path from 'path';

const NODE_DEBUG = ((<any>process.env).NODE_DEBUG || (<any>process.env).NODE_ENV === 'debug');

let moduleBasePath = Path.dirname(module.id);
if (Path.basename(moduleBasePath) == 'src') {
  moduleBasePath = Path.dirname(moduleBasePath);
  moduleBasePath = Path.join(moduleBasePath, 'build', 'js');
}

if (NODE_DEBUG && Fs.existsSync(Path.join(moduleBasePath, 'Debug'))) {
  java.classpath.push(Path.join(moduleBasePath, 'Debug', 'blazingsql-algebra.jar'));
  java.classpath.push(Path.join(moduleBasePath, 'Debug', 'blazingsql-algebra-core.jar'));
} else {
  java.classpath.push(Path.join(moduleBasePath, 'Release', 'blazingsql-algebra.jar'));
  java.classpath.push(Path.join(moduleBasePath, 'Release', 'blazingsql-algebra-core.jar'));
}

export const ArrayList = java.import('java.util.ArrayList');

export const CatalogColumnDataType =
  java.import('com.blazingdb.calcite.catalog.domain.CatalogColumnDataType');

export const CatalogColumnImpl =
  java.import('com.blazingdb.calcite.catalog.domain.CatalogColumnImpl');

export const CatalogTableImpl =
  java.import('com.blazingdb.calcite.catalog.domain.CatalogTableImpl');

export const CatalogDatabaseImpl =
  java.import('com.blazingdb.calcite.catalog.domain.CatalogDatabaseImpl');

export const BlazingSchema = java.import('com.blazingdb.calcite.schema.BlazingSchema');

export const RelConversionException =
  java.import('org.apache.calcite.tools.RelConversionException');

export const RelationalAlgebraGenerator =
  java.import('com.blazingdb.calcite.application.RelationalAlgebraGenerator');

export const SqlValidationException =
  java.import('com.blazingdb.calcite.application.SqlValidationException');

export const SqlSyntaxException =
  java.import('com.blazingdb.calcite.application.SqlSyntaxException');
