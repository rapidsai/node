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

import * as fs from 'fs';
import * as Module from 'module';
import * as Path from 'path';
import * as vm from 'vm';

import {createImport} from './import';
import {Resolver} from './resolve';
import {Transform} from './transform';

export interface Require extends NodeJS.Require {
  main: ContextModule;
}

/**
 * Creates a custom Module object which runs all required scripts in a provided vm context.
 */
export function createContextRequire(options: CreateContextRequireOptions) {
  return new ContextModule(options)._require;
}

let moduleId = 0;

/**
 * Patch nodejs module system to support context,
 * compilation and module resolution overrides.
 */

const Module_: any                   = getRealModule();
const module_static__load            = Module_._load;
const module_static__cache           = Module_._cache;
const module_static__extensions_js   = Module_._extensions['.js'];
const module_static__resolveFilename = Module_._resolveFilename;
const module_prototype__compile      = Module_.prototype._compile;
const module_prototype_load          = Module_.prototype.load;

export interface CreateContextRequireOptions {
  /** The directory from which to resolve requires for this module. */
  dir: string;
  /** A vm.Context to be used as the context for any required modules. */
  context: any;
  parent?: Module;
  /** A function to override the native module resolution. */
  resolve?: Resolver;
  /** A function to transform code during dynamic import. */
  transform?: Transform;
  /** An object containing any context specific require hooks to be used in this module. */
  extensions?: Partial<NodeJS.RequireExtensions>;
  /** A function to make the exports object for ES6 modules */
  makeExports?: (module: Module, exports: any) => any,
}

export class ContextModule extends Module {
  declare public _context: vm.Context;
  declare public _moduleCache: NodeJS.Dict<NodeModule>;
  declare public _resolveCache: NodeJS.Dict<string>;
  declare public _extensions?: Partial<NodeJS.RequireExtensions>;

  declare public _require: Require;
  declare public __resolve: Resolver;
  declare public _transform: Transform;

  declare public _exports: (mod: Module, exports: any)       => any;
  declare public _cachedDynamicImporter: (specifier: string) => Promise<vm.Module>;

  /**
   * Custom nodejs Module implementation which uses a provided
   * resolver, require hooks, and context.
   */
  constructor({
    parent = require.main ?? module,
    dir    = parent.path,
    context,
    resolve,
    extensions,
    makeExports = (_: Module, e: any) => e,
    transform = (_path: string, code: string) => code,
  }: CreateContextRequireOptions) {
    const filename = Path.join(dir, `index.${moduleId++}.ctx`);
    super(filename, parent);

    this.filename = filename;
    this.paths    = Module_._nodeModulePaths(dir);

    this._moduleCache  = {};
    this._resolveCache = {};
    this._transform    = transform;
    this._extensions   = extensions;
    this._exports      = makeExports;
    this._require      = createRequire(this);
    this._context      = isContext(context) ? context : vm.createContext(context);

    if (typeof resolve === 'function') {
      this.__resolve = resolve;
    } else {
      this._resolveFilename = module_static__resolveFilename;
    }

    this._cachedDynamicImporter = createImport(
      this._require, context, (path: string, code: string) => this._transform(path, code));

    this.loaded = true;
  }

  public static _load(request: string, parent: Module, isMain: boolean) {
    return patched_static__load.call(ContextModule, request, parent, isMain);
  }

  public static _resolveFilename(request: string, parent: Module, isMain?: boolean, options?: {
    paths?: string[]
  }) {
    return patched_static__resolveFilename.call(ContextModule, request, parent, isMain, options);
  }

  public _resolveFilename(request: string, parent: Module, isMain?: boolean, options?: {
    paths?: string[]|undefined;
  }) {
    const cache    = this._resolveCache;
    const cacheKey = `${parent.path}\x00${request}`;
    if (cache[cacheKey]) { return cache[cacheKey]!; }
    const resolved = this.__resolve(request, parent, isMain, options);
    if (resolved) { cache[cacheKey] = resolved; }
    return resolved;
  }

  public _compile(content: string, filename: string) {
    return patched_prototype__compile.call(this, content, filename);
  }

  public load(filename: string) { return patched_prototype_load.call(this, filename); }

  public exec(require: Require, content: string, filename: string, inner = this._context) {
    const outer   = this._context;
    const options = {
      filename,
      displayErrors: true,
      importModuleDynamically: this.createDynamicImporter(require, inner),
    };
    return tryAndReset(
      () => { this._context = inner; },
      () => runScript(inner, new vm.Script(content, options)),
      () => { this._context = outer; },
    );
  }

  public createDynamicImporter(require: Require = this._require, context = this._context) {
    return context === this._context  //
             ? this._cachedDynamicImporter
             : createImport(require, context, this._transform);
  }
}

/**
 * Use custom require cache for context modules
 *
 * @param request The file to resolve.
 * @param parent The module requiring this file.
 * @param isMain
 */
function patched_static__load(request: string, parent: Module, isMain: boolean): any {
  if (Module_.builtinModules.indexOf(request) !== -1) {
    return module_static__load(request, parent, isMain);
  }

  const module = findNearestContextModule(parent);

  if (module) {
    const filename = patched_static__resolveFilename(request, parent, isMain);
    // ensure native addons are added to the global static Module_._cache
    const cache = Path.extname(filename) === '.node' ? module_static__cache : module._moduleCache;
    const child = cache[filename];
    if (child) {
      if (parent.children.indexOf(child) === -1) { parent.children.push(child); }
      return child.exports;
    }
    Module_._cache = cache;
  }

  return tryAndReset(
    () => {},
    () => module_static__load(request, parent, isMain),
    () => { Module_._cache = module_static__cache; },
  );
}

// compile `.js` files even if "type": "module" is set in the file's package.json
function patched_static__extensions_js(module: any, filename: string) {
  module._compile(fs.readFileSync(filename, 'utf8'), filename);
}

/**
 * Hijack native file resolution using closest custom resolver.
 *
 * @param request The file to resolve.
 * @param parent The module requiring this file.
 */
function patched_static__resolveFilename(
  request: string, parent: Module, isMain?: boolean, options?: {paths?: string[]|undefined;}) {
  if (Module_.builtinModules.indexOf(request) === -1) {
    const module = findNearestContextModule(parent);
    if (module) {
      return tryAndReset(
        () => { Module_._resolveFilename = module_static__resolveFilename; },
        () => module._resolveFilename(request, parent, isMain, options),
        () => { Module_._resolveFilename = patched_static__resolveFilename; },
      );
    }
  }
  return module_static__resolveFilename(request, parent, isMain, options) as string;
}

/**
 * Patch module.load to use the context's custom extensions if provided.
 *
 * @param filename
 */
function patched_prototype_load(this: Module, filename: string) {
  const module = findNearestContextModule(this) as ContextModule;
  if (module?._extensions) {
    const extension = Path.extname(filename);
    const compiler  = module._extensions[extension];
    if (compiler) {
      const original = Module_._extensions[extension];
      return tryAndReset(
        () => { Module_._extensions[extension] = compiler; },
        () => module_prototype_load.call(this, filename),
        () => { Module_._extensions[extension] = original; },
      );
    }
  }
  return module_prototype_load.call(this, filename);
}

/**
 * This overrides script compilation to ensure the nearest context module is used.
 *
 * @param content The file contents of the script.
 * @param filename The filename for the script.
 */
function patched_prototype__compile(this: Module, content: string, filename: string) {
  const module = findNearestContextModule(this);

  if (module) {
    const require = createRequire(this);
    const dirname = Path.dirname(filename);
    const exports = module._exports(this, this.exports);
    return module.exec(require, Module_.wrap(content), filename)
      .call(exports, exports, require, this, filename, dirname);
  }

  return module_prototype__compile.call(this, content, filename);
}

/**
 * Walks up a module tree to find the nearest context module.
 *
 * @param cur The starting module.
 */
function findNearestContextModule(mod: Module) {
  let cur: Module|null|undefined = mod;
  do {
    if (cur instanceof ContextModule) { return cur; }
    // eslint-disable-next-line no-cond-assign
  } while (cur = cur && cur.parent);
  return undefined;
}

/**
 * Helper which will run a vm script in a context.
 * Special case for JSDOM where `runVMScript` is used.
 *
 * @param context The vm context to run the script in (or a jsdom instance).
 * @param script The vm script to run.
 */
function runScript(context: vm.Context, script: vm.Script) {
  return context.runVMScript
           ? context.runVMScript(script)
           : script.runInContext(context.getInternalVMContext ? context.getInternalVMContext()
                                                              : context);
}

function isContext(context: any) {
  return vm.isContext(context) || (typeof context.runVMScript === 'function' &&
                                   typeof context.getInternalVMContext === 'function');
}

function tryAndReset<Work extends() => any>(setup: () => any, work: Work, reset: () => any) {
  setup();
  let result: ReturnType<Work>;
  try {
    result = work();
  } catch (e) {
    reset();
    throw e;
  }
  reset();
  return result;
}

/**
 * Creates a require function bound to a module
 * and adds a `resolve` function the same as nodejs.
 *
 * @param mod The module to create a require function for.
 */
function createRequire(mod: Module, main = findNearestContextModule(mod)) {
  function require(id: string) {
    return tryAndReset(
      installModuleHooks,
      () => mod.require(id),
      uninstallModuleHooks,
    );
  }
  function resolve(id: string, options?: {paths?: string[]|undefined;}) {
    return patched_static__resolveFilename(id, mod, false, options);
  }
  require.main       = main;
  require.cache      = main && main._moduleCache;
  require.resolve    = resolve as NodeJS.RequireResolve;
  require.extensions = main && main._extensions || Module_._extensions;
  return require as Require;
}

let installedCount = 0;

// Jest hijacks the Module builtin, this gets the real one.
function getRealModule(): typeof Module {
  const M = Module as any;
  return M.__proto__ && M.__proto__.prototype ? M.__proto__ : M;
}

function installModuleHooks() {
  if (++installedCount === 1) {
    Module_._load              = patched_static__load;
    Module_._extensions['.js'] = patched_static__extensions_js;
    Module_._resolveFilename   = patched_static__resolveFilename;
    Module_.prototype._compile = patched_prototype__compile;
    Module_.prototype.load     = patched_prototype_load;
  }
}

function uninstallModuleHooks() {
  if (--installedCount === 0) {
    Module_._load              = module_static__load;
    Module_._extensions['.js'] = module_static__extensions_js;
    Module_._resolveFilename   = module_static__resolveFilename;
    Module_.prototype._compile = module_prototype__compile;
    Module_.prototype.load     = module_prototype_load;
  }
}
