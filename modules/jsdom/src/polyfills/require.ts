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
import * as path from 'path';
import * as vm from 'vm';

/**
 * Creates a custom Module object which runs all required scripts in a provided vm context.
 */
export function createContextRequire(options: CreateContextRequireOptions) {
  const mod = new ContextModule(options);
  return createRequire(mod)('esm')(mod, {
    'cjs': true,
    'wasm': true,
    'debug': true,
    'force': true,
    'mode': 'auto',
    'cache': false,
    'sourceMap': true,
    'mainFields': ['main', 'module', 'esnext'],
    // 'mainFields': ['esnext', 'module', 'main'],
    // 'mainFields': ['esnext', 'module', 'main'].reverse()
  });
}

let moduleId = 0;

/**
 * Patch nodejs module system to support context,
 * compilation and module resolution overrides.
 */
const Module_: any                   = Module;
const module_static__load            = Module_._load;
const module_static__resolveFilename = Module_._resolveFilename;
const module_prototype__compile      = Module_.prototype._compile;
const module_prototype_load          = Module_.prototype.load;

interface CreateContextRequireOptions {
  /** The directory from which to resolve requires for this module. */
  dir: string;
  /** A vm context which will be used as the context for any required modules. */
  context: any;
  /** A function to which will override the native module resolution. */
  resolve?: (from: string, request: string) => string;
  /** An object containing any context specific require hooks to be used in this module. */
  extensions?: NodeJS.RequireExtensions;
}

class ContextModule extends Module {
  declare public _context: any;
  declare public _cache: any;
  declare public _relativeResolveCache: Record<string, string>;
  declare public _resolve?: (from: string, request: string) => string;
  declare public _hooks?: NodeJS.RequireExtensions;

  /**
   * Custom nodejs Module implementation which uses a provided
   * resolver, require hooks, and context.
   */
  constructor({dir, context, resolve, extensions}: CreateContextRequireOptions) {
    const filename = path.join(dir, `index.${moduleId++}.ctx`);
    super(filename, require.main);

    this.filename              = filename;
    this._context              = context;
    this._resolve              = resolve;
    this._hooks                = extensions;
    this._cache                = {};
    this._relativeResolveCache = {};
    this.paths                 = Module_._nodeModulePaths(dir);

    if (!vm.isContext(context) && typeof context.runVMScript !== 'function' &&
        typeof context.getInternalVMContext !== 'function') {
      vm.createContext(context);
    }
  }

  public static _load(request: string, parent: Module, isMain: boolean) {
    return patched_static__load.call(ContextModule, request, parent, isMain);
  }

  public static _resolveFilename(request: string, parent: Module, isMain?: boolean, options?: {
    paths?: string[]
  }) {
    return patched_static__resolveFilename.call(ContextModule, request, parent, isMain, options);
  }

  public _compile(content: string, filename: string) {
    return patched_prototype__compile.call(this, content, filename);
  }

  public load(filename: string) { return patched_prototype_load.call(this, filename); }

  declare private _vm: typeof import('vm');

  public get vm() { return this._vm || (this._vm = patchVMForContextModule(this)); }
}

/**
 * Use custom require cache for context modules
 *
 * @param request The file to resolve.
 * @param parent The module requiring this file.
 * @param isMain
 */
function patched_static__load(request: string, parent: Module, isMain: boolean): any {
  if (request === 'vm') {
    return findNearestContextModule(parent)?.vm || module_static__load(request, parent, isMain);
  }

  if (Module.builtinModules.indexOf(request) !== -1) {
    return module_static__load(request, parent, isMain);
  }

  const previousCache = Module_._cache;
  const contextModule = findNearestContextModule(parent);

  if (contextModule) {
    const filename = patched_static__resolveFilename(request, parent, isMain);
    const cached   = contextModule._cache[filename];
    if (cached) {
      if (parent.children.indexOf(cached) === -1) {  //
        parent.children.push(cached);
      }
      return cached.exports;
    }
    Module_._cache = contextModule._cache;
  }

  return tryAndReset(
    () => {},
    () => module_static__load(request, parent, isMain),
    () => { Module_._cache = previousCache; },
  );
}

/**
 * Hijack native file resolution using closest custom resolver.
 *
 * @param request The file to resolve.
 * @param parent The module requiring this file.
 */
function patched_static__resolveFilename(
  request: string, parent: Module, isMain?: boolean, options?: {paths?: string[]|undefined;}):
  string {
  const isNotBuiltin  = Module.builtinModules.indexOf(request) === -1;
  const contextModule = isNotBuiltin && findNearestContextModule(parent);

  if (contextModule) {
    const resolver = contextModule._resolve;

    if (resolver) {
      // Normalize paths for custom resolvers.
      const dir = path.dirname(parent.filename);

      if (path.isAbsolute(request)) {
        request = path.relative(dir, request);

        if (request[0] !== '.') { request = './' + request; }
      }

      const relResolveCacheKey = `${dir}\x00${request}`;
      return (contextModule._relativeResolveCache[relResolveCacheKey] ||
              (contextModule._relativeResolveCache[relResolveCacheKey] = resolver(dir, request)));
    } else {
      return module_static__resolveFilename(request, parent, isMain, options);
    }
  }

  return module_static__resolveFilename(request, parent, isMain, options);
}

/**
 * Patch module.load to use the context's custom extensions if provided.
 *
 * @param filename
 */
function patched_prototype_load(this: NodeModule, filename: string) {
  const contextModule = findNearestContextModule(this) as ContextModule;
  if (contextModule) {
    const extensions = contextModule._hooks;
    const ext        = path.extname(filename);
    const compiler   = extensions && extensions[ext];
    if (compiler) {
      const original = Module_._extensions[ext];
      return tryAndReset(
        () => { Module_._extensions[ext] = compiler; },
        () => module_prototype_load.call(this, filename),
        () => { Module_._extensions[ext] = original; },
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
  const contextModule = findNearestContextModule(this);

  if (contextModule) {
    const code = Module.wrap(content);
    const opts = {filename, lineOffset: 0, displayErrors: true};
    const init = runScript(contextModule._context, new vm.Script(code, opts));
    return init.call(
      this.exports,
      this.exports,
      createRequire(this),
      this,
      filename,
      path.dirname(filename),
    );
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
function runScript(context: any, script: vm.Script) {
  return context.runVMScript
           ? context.runVMScript(script)
           : script.runInContext(context.getInternalVMContext ? context.getInternalVMContext()
                                                              : context);
}

function tryAndReset(setup: () => any, work: () => any, reset: () => any) {
  setup();
  let result = undefined;
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
function createRequire(mod: Module): NodeJS.Require {
  const main = findNearestContextModule(mod);
  function require(id: string) {
    return tryAndReset(
      () => { installModuleHooks(Module); },
      () => mod.require(id),
      () => { uninstallModuleHooks(Module); },
    );
  }
  function resolve(id: string, options?: {paths?: string[]|undefined;}) {
    return patched_static__resolveFilename(id, mod, false, options);
  }
  require.main       = main;
  require.cache      = main && main._cache;
  require.resolve    = resolve as NodeJS.RequireResolve;
  require.extensions = main && main._hooks || Module_._extensions;
  return require;
}

let installedCount = 0;

function installModuleHooks(Module: any) {
  if (++installedCount === 1) {
    // Jest hijacks the Module builtin, this gets the real one.
    Module       = Module.__proto__ && Module.__proto__.prototype ? Module.__proto__ : Module;
    Module._load = patched_static__load;
    Module._resolveFilename   = patched_static__resolveFilename;
    Module.prototype._compile = patched_prototype__compile;
    Module.prototype.load     = patched_prototype_load;
  }
}

function uninstallModuleHooks(Module: any) {
  if (--installedCount === 0) {
    // Jest hijacks the Module builtin, this gets the real one.
    Module       = Module.__proto__ && Module.__proto__.prototype ? Module.__proto__ : Module;
    Module._load = module_static__load;
    Module._resolveFilename   = module_static__resolveFilename;
    Module.prototype._compile = module_prototype__compile;
    Module.prototype.load     = module_prototype_load;
  }
}

function patchVMForContextModule(mod: ContextModule) {
  const clone = (obj: any) => Object.create(Object.getPrototypeOf(obj),  //
                                            Object.getOwnPropertyDescriptors(obj));

  function Script_(this: any, ...args: ConstructorParameters<typeof vm.Script>) {
    return Reflect.construct(vm.Script, args, new.target || Script_);
  }

  Script_.prototype = clone(vm.Script.prototype);

  Script_.prototype.runInNewContext = function(context?: vm.Context,
                                               options?: vm.RunningScriptOptions) {
    if (context?.global?.window === mod._context.window) {
      context = Object.create(Object.getPrototypeOf(mod._context), {
        ...Object.getOwnPropertyDescriptors(mod._context),
        ...Object.getOwnPropertyDescriptors(context),
      });
    }
    return vm.Script.prototype.runInNewContext.call(this, context, options);
  };

  Script_.prototype.runInThisContext = function(options?: vm.RunningScriptOptions) {
    return vm.Script.prototype.runInNewContext.call(this, mod._context, options);
  };

  const vm_ = Object.assign(clone(vm), {Script: Script_});

  vm_.compileFunction = function(
    code: string, params?: readonly string[], options?: vm.CompileFunctionOptions) {
    if (!options?.parsingContext) {
      return vm.compileFunction.call(
        this, code, params, {...options, parsingContext: mod._context});
    }
    return vm.compileFunction.call(this, code, params, options);
  };

  return vm_;
}
