import * as fs from "fs";
import * as path from "path";
import * as jsdom from 'jsdom';
import createContextRequire, {
    Types as TContextRequire
} from "./context-require";

const browserResolve = require('lasso-resolve-from');

export namespace Types {
    export interface Options extends jsdom.ConstructorOptions {
        /** The directory from which to resolve requires for this module. */
        dir: string;
        /** The initial html to parse with jsdom. */
        html?: string;
        /** An object containing any browser specific require hooks to be used in this module. */
        extensions?: TContextRequire.Hooks;
        /** A function called with the window, and the module, before parsing html. */
        beforeParse?(window: Window, context?: jsdom.JSDOM): void;
    }

    export interface JSDOMModule extends jsdom.JSDOM {
        require: TContextRequire.RequireFunction;
        window: jsdom.DOMWindow & { [key: string]: any };
    }
}

// Expose module.
module.exports = exports = createJSDOMContextRequire;
export default createJSDOMContextRequire;

/**
 * Creates a custom Module object which runs all required scripts
 * in a new jsdom instance.
 */
function createJSDOMContextRequire(options: Types.Options): Types.JSDOMModule {
    const { html, dir, extensions, beforeParse, ...jsdomOptions } = options;
    const context = new jsdom.JSDOM("", { runScripts: "dangerously", ...jsdomOptions }) as Types.JSDOMModule;
    const { window } = context;
    const resolveConfig = {
        remaps: loadRemaps,
        extensions:
            extensions &&
            ([] as string[])
                .concat(Object.keys(require.extensions))
                .concat(Object.keys(extensions))
                .filter(unique)
    };

    context.require = createContextRequire({ dir, context, resolve, extensions });

    // Pass through istanbul coverage.
    (window as any).__coverage__ = (global as any).__coverage__;

    if (beforeParse) {
        beforeParse(window, context);
    }

    window.document.open();
    window.document.write(
        html || "<!DOCTYPE html><html><head></head><body></body></html>"
    );

    return context;

    /**
     * A function to resolve modules in the browser using the provided config.
     *
     * @param from The file being resolved from.
     * @param request The requested path to resolve.
     */
    function resolve(from: string, request: string): string {
        let loc = browserResolve(from, request, resolveConfig);
        if (loc === undefined) {
            throw new Error(`Could not resolve module path '${request}' from dir '${from}'`);
        }
        return loc.path;
    }
}

/**
 * Array filter for uniqueness.
 */
function unique(item: any, i: number, list: any[]): boolean {
    return list.indexOf(item) === i;
}

const remapCache = Object.create(null);

/**
 * Loads browser.json remaps.
 *
 * @param dir The directory to load remaps from.
 */
function loadRemaps(dir: string) {
    const file = path.join(dir, "browser.json");

    if (file in remapCache) {
        return remapCache[file];
    }

    let result: Record<string, string> | undefined = undefined;
    const remaps = fs.existsSync(file) && require(file).requireRemap;

    if (remaps) {
        result = {};
        for (const remap of remaps) {
            result[path.join(dir, remap.from)] = path.join(dir, remap.to);
        }
    }

    remapCache[file] = result;
    return result;
}
