// Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

// import {test} from '@jest/globals';
// import {Device, devices} from '@rapidsai/cuda';

export const sizes = {
  '1_MiB': 1 << 20,
  '2_MiB': 1 << 21,
  '4_MiB': 1 << 23,
  '16_MiB': 1 << 24,
};

// export const testForEachDevice = (name: string, fn: (() => void)|((d: Device) => void)) =>
//   test.each([...devices])(name, (d: Device) => {
//     d.callInContext(() => {
//       d.synchronize();
//       fn(d);
//       d.synchronize();
//     });
//   });

import {
  beforeAll,
  afterAll,
  beforeEach,
  afterEach,
} from '@jest/globals';

/* eslint-disable @typescript-eslint/no-misused-promises */
/* eslint-disable @typescript-eslint/no-floating-promises */
beforeAll(flushStdout);
afterAll(flushStdout);
beforeEach(flushStdout);
afterEach(flushStdout);

function flushStdout() {
  return new Promise<void>((done) => {
    if (process.stdout.write('')) {
      done();
    } else {
      process.stdout.once('drain', () => { done(); });
    }
  });
}
