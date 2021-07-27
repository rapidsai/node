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

import {RapidsJSDOM} from '@rapidsai/jsdom';

test('nothing', () => {
  // stub test to make sure the debugger works
  debugger;
});

test('can require file inside customized JSDOM', () => {
  const {window} = (new RapidsJSDOM()).window;
  const success  = window.eval((function() {
    try {
      var f = require('./local_file');
      if (f) { return true; }
    } catch (e) {}
    return false;
  })().toString());

  const failure  = window.eval((function() {
    try {
      var f = require('./nonexistent_file');
      if (f) { return true; }
    } catch (e) {}
    return false;
  })().toString());

  expect(success).toBeTruthy();
  expect(failure).toBeFalsy();
});
