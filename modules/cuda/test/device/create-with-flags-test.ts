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

import { devices, Device, DeviceFlags } from '@nvidia/cuda';

test.each([
    DeviceFlags.scheduleAuto,
    DeviceFlags.scheduleSpin,
    DeviceFlags.scheduleYield,
    DeviceFlags.scheduleBlockingSync,
    DeviceFlags.lmemResizeToMax
])(`Creates each device with DeviceFlag %i`, (flags) => {
    for (const i of Array.from({ length: devices.length }, (_, i) => i)) {
        const device = new Device(i, flags);
        try {
            expect(device.id).toBeDefined();
            expect(device.getFlags()).toBe(flags);
        } finally {
            device.reset();
        }
    }
});
