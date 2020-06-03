import { devices } from '@nvidia/cuda';

test(`device.properties`, () => {
    for (const device of devices) {
        const { properties } = device;
        expect(properties).toBeDefined();
        expect(properties.name).toBeDefined();
    }
});
