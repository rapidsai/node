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

import * as babel from '@babel/core';
import * as fs from 'fs';
import * as Module from 'module';
import * as Path from 'path';

const cloneDeep        = require('clone-deep');
const sourceMapSupport = require('source-map-support');

export type Transform = (path: string, code: string) => string;
const identityTransform: Transform = (_: string, code: string) => code;

export function createTransform({preTransform = identityTransform, ...opts}:
                                  {preTransform?: Transform}&Partial<babel.TransformOptions> = {}) {
  let transform = preTransform;

  if (opts) {
    const maps: any         = {};
    const transformOpts     = normalizeOptions(opts);
    const installSourceMaps = supportSourceMaps(maps);
    transform = (path: string, code: string) => {
      const content = preTransform(path, code);
      // merge in base options and resolve all the plugins and presets relative to this file
      const compileOpts = babel.loadOptions({
        // sourceRoot can be overwritten
        sourceRoot: Path.dirname(path) + Path.sep,
        ...cloneDeep(transformOpts),
        filename: path
      });
      if (compileOpts) {
        const {sourceMaps = 'both'} = <any>compileOpts;
        const transformed = babel.transform(content, {...compileOpts, sourceMaps, ast: false});
        if (transformed) {
          if (transformed.map) {
            installSourceMaps();
            maps[path] = transformed.map;
          }
          return transformed.code || content;
        }
      }
      return content;
    };
  }

  return {transform, extensions: compilersByExtension(transform)};
}

function compilersByExtension(transform: Transform) {
  return {
    ...(Module as any)._extensions,
    // compile `.js` files even if "type": "module" is set in the file's package.json
    ['.js']: transformAndCompile,
    ['.jsx']: transformAndCompile,
    ['.mjs']: transformAndCompile,
    ['.cjs']: transformAndCompile,
  };

  function transformAndCompile(module: Module, filename: string) {
    return (<any>module)._compile(transform(filename, fs.readFileSync(filename, 'utf8')), filename);
  }
}

function normalizeOptions(options: Partial<babel.TransformOptions>) {
  const opts = {...options, caller: {name: '@rapidsai/jsdom', ...options.caller}};
  // Ensure that the working directory is resolved up front so that
  // things don't break if it changes later.
  opts.cwd = Path.resolve(opts.cwd || '.');
  if (opts.ignore === undefined && opts.only === undefined) {
    opts.only = [
      // Only compile things inside the current working directory.
      // $FlowIgnore
      new RegExp('^' + escapeRegExp(opts.cwd), 'i'),
    ];
    opts.ignore = [
      // Ignore any node_modules inside the current working directory.
      new RegExp(
        '^' +
          // $FlowIgnore
          escapeRegExp(opts.cwd) + '(?:' + Path.sep + '.*)?' +
          // $FlowIgnore
          escapeRegExp(Path.sep + 'node_modules' + Path.sep),
        'i',
        ),
    ];
  }

  return opts;

  function escapeRegExp(string: string) { return string.replace(/[|\\{}()[\]^$+*?.]/g, '\\$&'); }
}

function supportSourceMaps(maps: any) {
  let sourceMapsSupportInstalled  = false;
  return ()                      => {
    if (!sourceMapsSupportInstalled) {
      sourceMapSupport.install({
        handleUncaughtExceptions: false,
        environment: 'node',
        retrieveSourceMap(filename: string) {
          const map = maps && maps[filename];
          if (map) {
            return {
              url: null,
              map: map,
            };
          } else {
            return null;
          }
        },
      });
      sourceMapsSupportInstalled = true;
    }
  };
}
