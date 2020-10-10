import { devices, Device, DeviceFlag } from '@nvidia/cuda';

test.each([
    DeviceFlag.scheduleAuto,
    DeviceFlag.scheduleSpin,
    DeviceFlag.scheduleYield,
    DeviceFlag.scheduleBlockingSync,
    DeviceFlag.lmemResizeToMax
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
