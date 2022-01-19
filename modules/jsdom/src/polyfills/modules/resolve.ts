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

import * as Module from 'module';
import * as Path from 'path';

export type Resolver =
  (request: string, parent: Module, isMain?: boolean, options?: {paths?: string[]|undefined;}) =>
    string;

export type ResolversMap = {
  [path: string]: Resolver
};

const nodeModulesWithSeparator       = `node_modules${Path.sep}`;
const nodeModulesWithSeparatorLength = nodeModulesWithSeparator.length;

export function createResolve(resolvers?: ResolversMap): Resolver {
  resolvers = resolvers || {};

  return resolve;

  function resolve(request: string, parent: Module, isMain?: boolean, options?: any) {
    // Normalize request path so custom resolvers can have a stable key
    // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
    const resolveFilename = resolvers![resolverKey(request, parent)];
    return resolveFilename  //
             ? resolveFilename(request, parent, isMain, options)
             : (<any>Module)._resolveFilename(request, parent, isMain, options);
  }

  function resolverKey(request: string, parent: Module) {
    const dir = '' + parent.path;
    const idx = dir.lastIndexOf(nodeModulesWithSeparator) ?? -1;
    const pre = dir.slice(~idx && idx + nodeModulesWithSeparatorLength || 0);
    return Path.join(pre, normalizeReq(request, parent));
  }

  function normalizeReq(request: string, parent: Module) {
    if (Path.isAbsolute(request)) {
      const relative = Path.relative(parent.path, request);
      if (!relative.startsWith(`.${Path.sep}`)) {  //
        return `.${Path.sep}${relative}`;
      }
      return relative;
    }
    return request;
  }
}
