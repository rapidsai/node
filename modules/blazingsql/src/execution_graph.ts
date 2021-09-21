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

import {DataFrame, Series} from '@rapidsai/cudf';
import {ExecutionGraphWrapper} from './node_blazingsql';

export class ExecutionGraph {
  private executionGraphWrapper: ExecutionGraphWrapper|undefined;

  constructor(executionGraphWrapper?: ExecutionGraphWrapper) {
    this.executionGraphWrapper = executionGraphWrapper;
  }

  start(): void { this.executionGraphWrapper?.start(); }

  result() {
    if (this.executionGraphWrapper === undefined) return new DataFrame();
    const {names, tables: [table]} = this.executionGraphWrapper.result();
    return new DataFrame(names.reduce(
      (cols, name, i) => ({...cols, [name]: Series.new(table.getColumnByIndex(i))}), {}));
  }

  sendTo(ralId: number, messageId: string): ExecutionGraph {
    return new ExecutionGraph(this.executionGraphWrapper?.sendTo(ralId, messageId));
  }
}
