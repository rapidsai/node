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

const {fromEvent, EMPTY}                  = require('rxjs');
const {tap, groupBy, mergeMap, concatMap} = require('rxjs/operators');

const {Renderer} = require('./render');
const renderer   = new Renderer();

function render({props = {}, graph = {}, state = {}, event = [], frame = 0}) {
  return renderer.render(props, graph, state, event, frame);
}

fromEvent(process, 'message', (x) => x)
  .pipe(groupBy(({type}) => type))
  .pipe(mergeMap((group) => {
    switch (group.key) {
      case 'exit': return group.pipe(tap(({code}) => process.exit(code)));
      case 'render.request':
        return group                                                                //
          .pipe(concatMap(({data}) => render(data), ({id}, data) => ({id, data})))  //
          .pipe(tap((result) => process.send({...result, type: 'render.result'})));
    }
    return EMPTY;
  }))
  .subscribe();
