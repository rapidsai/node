import { devices, DeviceFlag } from '@nvidia/cuda';

test.each([
    DeviceFlag.scheduleAuto,
    DeviceFlag.scheduleSpin,
    DeviceFlag.scheduleYield,
    DeviceFlag.scheduleBlockingSync,
    DeviceFlag.lmemResizeToMax
])(`Sets device flags to DeviceFlag %i`, (flags) => {
    for (const device of devices) {
        try {
            expect(device.id).toBeDefined();
            expect(device.getFlags()).toBe(DeviceFlag.scheduleAuto);
            const result = device.reset(flags).getFlags();
            expect(result).toBe(flags);
        } finally {
            device.reset();
        }
    }
});
