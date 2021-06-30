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

import {Context} from './context';

export class BlazingContext {
  public context: Context;  // TODO fix this

  constructor() {
    this.context = new Context({
      ralId: 0,
      workderId: '',
      network_iface_name: '',
      ralCommunicationPort: 0,
      workersUcpInfo: new Array(0),  // TODO: Fix.
      singleNode: false,
      allocationMode: 'default',
      initialPoolSize: null,
      maximumPoolSize: null,
      enableLogging: false,
    });
  }
}
