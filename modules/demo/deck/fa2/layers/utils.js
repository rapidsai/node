import GL from '@luma.gl/constants';
// import GL from 'luma.gl/constants';

/* deck.gl could potentially make this as a Layer utility? */
export function getLayerAttributes(LayerClass) {
    const layer = new LayerClass({});
    try {
        layer.context = {
            // gl: {
            //     __proto__: (WebGL2RenderingContext || WebGLRenderingContext).prototype,
            //     TEXTURE_BINDING_3D: 0x806a
            // }
        };
        layer._initState();
        layer.initializeState();
    } catch (error) {
        // ignore
    }
    const attributes = { ...layer.getAttributeManager().getAttributes() };

    for (const attributeName in attributes) {
        attributes[attributeName] = Object.assign({}, {
            offset: attributes[attributeName].settings.offset,
            stride: attributes[attributeName].settings.stride,
            type: attributes[attributeName].settings.type || GL.FLOAT,
            size: attributes[attributeName].settings.size,
            divisor: attributes[attributeName].settings.divisor,
            normalize: attributes[attributeName].settings.normalized,
            integer: attributes[attributeName].settings.integer,
        });
        // attributes[attributeName].type = attributes[attributeName].settings.type || GL.FLOAT;
    }

    return attributes;
}
