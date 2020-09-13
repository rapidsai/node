// Copyright (c) 2020, NVIDIA CORPORATION.
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

import * as Path from 'path';
import * as jsdom from 'jsdom';
import { ImageData } from 'canvas';
import { tmpdir } from './object-url';
import { parse as parseURL } from 'url';
import { GLFWDOMWindowOptions } from './window';
import createContextRequire from "context-require";
import { JSDOM_KEYS, NODE_GLOBAL_KEYS } from './keys';
import createJSDOMContextRequire, { Types } from './require';

const jsdomGlobal = require('jsdom-global');
const browserResolve = require('lasso-resolve-from');
const idlUtils = require('jsdom/lib/jsdom/living/generated/utils');
const localhostUrl = `http://${Path.basename(tmpdir)}/`.toLowerCase();

class ImageLoader extends jsdom.ResourceLoader {
    fetch(url: string, options: jsdom.FetchOptions) {
        // Hack since JSDOM 16.2.2: If loading a relative file
        // from our dummy localhost URI, translate to a file:// URI.
        if (url.startsWith(localhostUrl)) {
            url = url.slice(localhostUrl.length);
        }
        // url.endsWith('/') && (url = url.slice(0, -1));
        const isDataURI = url && url.startsWith('data:');
        const isFilePath = !isDataURI && !parseURL(url).protocol;
        return super.fetch(isFilePath ? `file://${url}` : url, options);
    }
}

const jsdomOptions = {
    url: localhostUrl,
    // probably need this
    pretendToBeVisual: true,
    // make event dispatching work
    runScripts: 'dangerously',
    // load local resources
    resources: new ImageLoader(),
};

function createJSDOMContext(dir = process.cwd(), runInThisContext = false, code = '') {
    let context: any;
    const processClone = Object.create(process, {
        browser: { value: true },
        type:  { value: 'renderer' }
    });
    if (runInThisContext) {
        if (!(global as any).window) {
            (global as any).idlUtils = idlUtils;
            (global as any).ImageData = ImageData;
            (global as any).uninstallJSDOM = jsdomGlobal(undefined, { ...jsdomOptions });
        }
        context = Object.create(global, {
            window: { value: undefined, writable: true, configurable: true },
            require: { value: undefined, writable: true, configurable: true },
        });
        context.window = Object.create((global as any).window);
        context.require = createContextRequire({ dir, context, resolve: browserResolve });
        context.window.openGLFWWindow = eval(`(() => ${wrapScriptInOpenWindowFn(code)})()`);
        JSDOM_KEYS.forEach((key) => { try { (context as any)[key] = context.window[key]; } catch (e) {} });
        NODE_GLOBAL_KEYS.forEach((key) => { try { (context.window as any)[key] = global[key]; } catch (e) {} });
    } else {
        context = createJSDOMContextRequire(<any> {
            dir, html: scriptToHTML(code), ...jsdomOptions
        });
        Object.assign(context, { ...global, global: context });
        Object.assign(context.window, { ...global, global: context });
        JSDOM_KEYS.forEach((key) => { try { (context as any)[key] = context.window[key]; } catch (e) {} });
        NODE_GLOBAL_KEYS.forEach((key) => { try { (context.window as any)[key] = global[key]; } catch (e) {} });
    }
    Object.assign(context, { ImageData, process: processClone, global: context, idlUtils });
    Object.assign(context.window, { ImageData, process: processClone, require: context.require });
    return context;
}

const wrapScriptInOpenWindowFn = (code: string) => `
function openGLFWWindow(opts = {}) {
    try {
        require('${__dirname}/globals')(Object.assign({}, opts));
        (${code})(Object.assign({}, opts));
    } catch (e) {
        console.error(e && (e.stack || e.message) || \`\${e}\`);
        process.exit(1);
    }
}`;
const scriptToHTML = (code: string) => `
<!DOCTYPE html><html><head><meta charset="utf-8"></head>
<body><script type="text/javascript">${wrapScriptInOpenWindowFn(code)}</script></body></html>`;

export function createWindow(code: Function | string, runInThisContext = false) {
    const context = createJSDOMContext(process.cwd(), runInThisContext, code.toString());
    context.open = (opts: GLFWDOMWindowOptions = {}) => context.window.openGLFWWindow(opts);
    return context as (Types.JSDOMModule & { open(options?: GLFWDOMWindowOptions): void; });
}

export function createModuleWindow(id: string, runInThisContext = false) {
    return createWindow(`function() { return require('${id}'); }`, runInThisContext);
}

export function createReactWindow(id: string, runInThisContext = false) {
    return createWindow(`function (props) {
            var Component = require('${id}');
            var reactDOM = require('react-dom');
            var createElement = require('react').createElement;
            var render = reactDOM.render, FDN = reactDOM.findDOMNode;
            props.ref || (props.ref = e => window._inputEventTarget = FDN(e));
            render(
                createElement(Component.default || Component, props),
                document.body.appendChild(document.createElement('div'))
            );
        }`, runInThisContext);
}

process.on(<any> 'uncaughtException', (err: Error, origin: any) => {
    process.stderr.write(
        `Exception origin: ${origin}\n` +
        `Caught exception: ${err && err.stack || err}\n`
    );
});
