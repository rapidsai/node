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

export * from './loadnativemodule';

export const modules_path = Path.resolve(__dirname, '..', '..', '..');

export const project_root_dir_path = Path.resolve(modules_path, '..');

export const sccache_path = Path.resolve(modules_path, '.cache', 'sccache');

export const cpm_source_cache_path = Path.resolve(modules_path, '.cache', 'cpm');

export const cpm_binary_cache_path = Path.resolve(modules_path, '.cache', 'build');

export const cpp_core_include_path = Path.resolve(modules_path, 'core', 'include');

export const cmake_modules_path = Path.resolve(modules_path, 'core', 'cmake', 'Modules');
