import { devices } from '@nvidia/cuda';

test(`device.properties`, () => {
    for (const device of devices) {
        const props = device.getProperties();
        expect(props).toBeDefined();
        expect(props.name).toBeDefined();
        expect(props.name).toBe(device.name);
    }
});
