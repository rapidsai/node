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

import {mkdtempSync, unlinkSync, writeFileSync} from 'fs';
import * as jsdom from 'jsdom';
import * as os from 'os';
import * as Path from 'path';

export function createObjectUrlAndTmpDir() {
  const tmpdir = mkdtempSync(os.tmpdir() + Path.sep);
  const url    = `http://${Path.basename(tmpdir)}/`.toLowerCase();
  return {url, tmpdir};
}

export function installObjectURL(tmpdir: string) {
  return function installObjectURL(window: jsdom.DOMWindow) {
    let filesCount = 0;
    const map      = new Map<URL, string>();

    window.jsdom.global.URL ??= window.URL;

    [window.URL, window.jsdom.global.URL].forEach((URL) => {
      URL.createObjectURL ??= createObjectURL;
      URL.revokeObjectURL ??= revokeObjectURL;
    });

    return window;

    function createObjectURL(blob: Blob) {
      const path = Path.join(tmpdir, `${filesCount++}`);
      writeFileSync(path, window.jsdom.utils.implForWrapper(blob)._buffer);
      const url = new window.jsdom.global.URL(`file://${path}`);
      map.set(url, path);
      return url;
    }

    function revokeObjectURL(url: any) {
      if (map.has(url)) {
        // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
        const p = map.get(url)!;
        map.delete(url);
        unlinkSync(p);
      }
    }
  };
}
