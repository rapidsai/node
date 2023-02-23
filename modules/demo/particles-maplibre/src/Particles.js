import React, {useEffect, useRef} from 'react';
import regl from 'regl';

function Particles() {
  const canvasRef = useRef(null);

  useEffect(() => {
    const reglInstance = regl(canvasRef.current);
    const drawTriangle = reglInstance({
      frag: `
        precision mediump float;
        uniform vec4 color;
        void main() {
          gl_FragColor = color;
        }
      `,
      vert: `
        attribute vec2 position;
        void main() {
          gl_Position = vec4(position, 0, 1);
        }
      `,
      uniforms: {
        color: [1, 0, 0, 0.5],
      },
      attributes: {
        position: [
          [-0.5, -0.5],
          [0, 1],
          [1, -1],
        ],
      },
      count: 3,
    });

    drawTriangle();

    return () => reglInstance.destroy();
  }, []);

  return <canvas ref = {canvasRef} className = 'foreground-canvas' />;
}

export default Particles;
