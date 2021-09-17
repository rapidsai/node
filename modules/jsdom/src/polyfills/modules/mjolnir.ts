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

export function mjolnirHammerResolvers() {
  return {
    'mjolnir.js/src/utils/hammer': mjolnirHammerResolver,
    'mjolnir.js/dist/es5/utils/hammer': mjolnirHammerResolver,
    'mjolnir.js/dist/esm/utils/hammer': mjolnirHammerResolver,
  };
}

function mjolnirHammerResolver(request: string, ...args: any[]) {
  request = request.replace('hammer', 'hammer.browser');
  return (Module as any)._resolveFilename(request, ...args);
}

// export function installMjolnirHammer(window: jsdom.DOMWindow) {
//   window.evalFn(async function() {
//     const hammerjs = require('hammerjs');
//     const {enhancePointerEventInput, enhanceMouseInput} =
//       require('mjolnir.js/dist/es5/utils/hammer-overrides');
//     enhancePointerEventInput(hammerjs.PointerEventInput);
//     enhanceMouseInput(hammerjs.MouseInput);
//     (await Promise.all([
//       import(`mjolnir.js/src/utils/hammer${'.js'}`),
//       import(`mjolnir.js/dist/es5/utils/hammer${'.js'}`),
//       import(`mjolnir.js/dist/esm/utils/hammer${'.js'}`),
//     ]))
//       .forEach((mjolnirHammer) => {
//         try {
//           redefine(mjolnirHammer, [
//             {field: 'default', value: hammerjs},
//             {field: 'Manager', value: hammerjs['Manager']},
//           ]);
//         } catch (e) { /**/
//         }
//       });
//     function redefine(target: any, fields: {field: string, value: any}[]) {
//       Object.defineProperties(target, fields.reduce((descriptors, {field, value}) => {
//         descriptors[field] = {...Object.getOwnPropertyDescriptor(target, field), value};
//         return descriptors;
//       }, {} as PropertyDescriptorMap));
//     }
//   });

//   return window;
// }
