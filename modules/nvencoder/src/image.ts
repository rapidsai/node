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

import NVENCODER from './addon';

export const rgbaMirror: (width: number, height: number, axis: number, source: any, target?: any) => void = NVENCODER.image.rgbaMirror;
export const bgraToYCrCb420: (width: number, height: number, source: any, target: any) => void = NVENCODER.image.bgraToYCrCb420;
