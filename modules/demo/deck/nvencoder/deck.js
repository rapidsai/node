import { Deck } from '@deck.gl/core';
import { GeoJsonLayer } from '@deck.gl/layers';
import { Framebuffer, Texture2D } from '@luma.gl/webgl';
import { createDeckGLVideoEncoderStream } from '@nvidia/deck.gl';

import sfZipcodes from './sf.zip.geo.json';

let numFrames = 0, framebuffer;

const deck = new Deck({
    _animate: true,
    controller: true,
    initialViewState: {
        longitude: -122.45,
        latitude: 37.76,
        zoom: 11,
        bearing: 0,
        pitch: 30
    },
    onWebGLInitialized(gl) {
        deck.setProps({
            _framebuffer: new Framebuffer(gl, {
                color: new Texture2D(gl, {
                    mipmaps: false,
                    parameters: {
                        [gl.TEXTURE_MIN_FILTER]: gl.LINEAR,
                        [gl.TEXTURE_MAG_FILTER]: gl.LINEAR,
                        [gl.TEXTURE_WRAP_S]: gl.CLAMP_TO_EDGE,
                        [gl.TEXTURE_WRAP_T]: gl.CLAMP_TO_EDGE
                    }
                })
            })
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
    onAfterRender() {
        if (deck.props._framebuffer) {
            framebuffer = deck.props._framebuffer;
            deck.setProps({ _framebuffer: null });
            deck.redraw(true);
        } else {
            deck.setProps({ _framebuffer: framebuffer });
            framebuffer = null;
            if (++numFrames >= 1000) {
                deck.setProps({ onAfterRender: () => {}});
                deck.finalize();
                setTimeout(() => process.exit(0), 20);
            }
        }
    }
});

createDeckGLVideoEncoderStream(deck)
    .then((outputs) => outputs.pipe(process.stdout, { end: false }));
