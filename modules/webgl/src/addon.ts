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

import { isElectron } from './iselectron';

export const gl = (() => {
    let gl: any, types = ['Release'];
    let name = `node_webgl${isElectron() ? '_electron' : ''}`;
    if (process.env.NODE_DEBUG !== undefined || process.env.NODE_ENV === 'debug') {
        types.push('Debug');
    }
    for (let type; type = types.pop();) {
        try {
            if (gl = require(`../${type}/${name}.node`)) {
                break;
            }
        } catch (e) { console.error(e); continue; }
    }
    if (gl) return gl;
    throw new Error('node_webgl not found');
})();

export default gl;
