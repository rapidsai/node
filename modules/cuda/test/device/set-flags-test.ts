import { devices, CUDADeviceFlag } from '@nvidia/cuda';

test.each([
    CUDADeviceFlag.scheduleAuto,
    CUDADeviceFlag.scheduleSpin,
    CUDADeviceFlag.scheduleYield,
    CUDADeviceFlag.scheduleBlockingSync,
    CUDADeviceFlag.lmemResizeToMax
])(`Sets device flags to CUDADeviceFlag %i`, (flags) => {
    for (const device of devices) {
        try {
            expect(device.id).toBeDefined();
            expect(device.getFlags()).toBe(CUDADeviceFlag.scheduleAuto);
            const result = device.reset(flags).getFlags();
            expect(result).toBe(flags);
        } finally {
            device.reset();
        }
    }
});
