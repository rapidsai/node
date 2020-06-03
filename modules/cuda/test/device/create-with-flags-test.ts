import { devices, CUDADevice, CUDADeviceFlag } from '@nvidia/cuda';

test.each([
    CUDADeviceFlag.scheduleAuto,
    CUDADeviceFlag.scheduleSpin,
    CUDADeviceFlag.scheduleYield,
    CUDADeviceFlag.scheduleBlockingSync,
    CUDADeviceFlag.lmemResizeToMax
])(`Creates each device with CUDADeviceFlag %i`, (flags) => {
    for (const i of Array.from({ length: devices.length }, (_, i) => i)) {
        const device = CUDADevice.new(i, flags);
        try {
            expect(device.id).toBeDefined();
            expect(device.getFlags()).toBe(flags);
        } finally {
            device.reset();
        }
    }
});
