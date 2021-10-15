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

import {glfw, GLFW, GLFWmonitor, Monitor} from '@rapidsai/glfw';
import {map, publish, refCount} from 'rxjs/operators';

import {glfwCallbackAsObservable, GLFWEvent} from './event';

export function monitorEvents() {
  return glfwCallbackAsObservable(glfw.setMonitorCallback)
    .pipe(map(([_, ...rest]) => GLFWMonitorEvent.create(_, ...rest)))
    .pipe(publish(), refCount());
}

class GLFWMonitorEvent extends GLFWEvent {
  public static create(monitor: GLFWmonitor, event: number) {
    const evt =
      new GLFWMonitorEvent(event === GLFW.CONNECTED ? 'monitorconnected' : 'monitordisconnected');
    evt.target = new Monitor(monitor, event === GLFW.CONNECTED);
    return evt;
  }
}
