import { Deck } from '@deck.gl/core';
import { GeoJsonLayer } from '@deck.gl/layers';
import sfZipcodes from './sf.zip.geo.json';

export default function createDeckInstance({
    createEncoderTarget,
    destroyEncoderTarget,
    onResize,
    onAfterRender,
    ...deckProps
}) {

    let numFrames = 0;
    let framebuffer;

    const deck = new Deck({
        ...deckProps,
        _animate: true,
        controller: true,
        initialViewState: {
            longitude: -122.45,
            latitude: 37.76,
            zoom: 11,
            bearing: 0,
            pitch: 30
        },
        onResize,
        onWebGLInitialized(gl) {
            deck.setProps({
                _framebuffer: framebuffer = createEncoderTarget(gl)
            });
        },
        onLoad() {
            deck.setProps({
                layers: [
                    new GeoJsonLayer({
                        data: sfZipcodes,
                        opacity: 0.5,
                        extruded: true,
                        getFillColor: [255, 0, 0],
                        getElevation: d => Math.random() * 3000
                    })
                ]
            });
        },
        onAfterRender(props) {
            if (deck.props._framebuffer) {
                onAfterRender(props);
                deck.setProps({ _framebuffer: null });
                deck.redraw(true);
            } else {
                deck.setProps({ _framebuffer: framebuffer });
                if (++numFrames >= 1000) {
                    destroyEncoderTarget();
                    setTimeout(() => process.exit(0), 20);
                    deck.setProps({ onAfterRender: () => {} });
                }
            }
        }
    });

    return deck;
}
