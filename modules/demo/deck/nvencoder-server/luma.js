import { AnimationLoop, Texture2D, setParameters } from '@luma.gl/core';
import { Matrix4 } from 'math.gl';
import { Star } from './star';

import { createAnimationLoopVideoEncoderStream } from '@nvidia/deck.gl';

export default function createAnimationLoopInstance(encoderOptions) {

    let zoom = -15;
    let tilt = 90;
    
    function keyboardEventHandler(e) {
        switch (e.code) {
            case 'ArrowUp':
                tilt -= 1.5;
                break;
            case 'ArrowDown':
                tilt += 1.5;
                break;
            case 'PageUp':
                zoom -= 0.1;
                break;
            case 'PageDown':
                zoom += 0.1;
                break;
            default:
        }
    }

    const loop = new AnimationLoop({
        createFramebuffer: true,
        onInitialize({ gl }) {

            document.addEventListener('keydown', keyboardEventHandler);
            document.addEventListener('keypress', keyboardEventHandler);

            setParameters(gl, {
                clearColor: [0, 0, 0, 1],
                clearDepth: 1,
                blendFunc: [gl.SRC_ALPHA, gl.ONE],
                blend: true
            });

            const texture = new Texture2D(gl, 'star.gif');

            const stars = [];
            const numStars = 50;
            for (let i = 0; i < numStars; i++) {
                stars.push(
                    new Star(gl, {
                        startingDistance: (i / numStars) * 5.0,
                        rotationSpeed: i / numStars,
                        texture
                    })
                );
            }

            return { stars };
        },
        onRender({ gl, aspect, stars, framebuffer }) {
            // Update Camera Position
            const radTilt = (tilt / 180) * Math.PI;
            const cameraY = Math.cos(radTilt) * zoom;
            const cameraZ = Math.sin(radTilt) * zoom;

            gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
            if (framebuffer) {
                framebuffer.clear({ color: true, depth: true });
            }

            for (const i in stars) {
                const uMVMatrix = new Matrix4()
                    .lookAt({ eye: [0, cameraY, cameraZ] })
                    .multiplyRight(stars[i].matrix);

                stars[i].setUniforms({
                    uMVMatrix,
                    uPMatrix: new Matrix4().perspective({ aspect })
                });
                stars[i].draw();
                if (framebuffer) {
                    stars[i].draw({ framebuffer });
                }
                stars[i].animate();
            }
        },
        onFinalize() {
            document.removeEventListener('keydown', keyboardEventHandler);
            document.removeEventListener('keypress', keyboardEventHandler);
        }
    });

    return createAnimationLoopVideoEncoderStream(loop, encoderOptions).then((frames) => ({ loop, frames }));
}
