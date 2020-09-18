const Fastify = require('fastify');

import * as stream from 'stream';
import { createReadStream } from 'fs';
import { CUDADevice } from '@nvidia/cuda';
import createAnimationLoopInstance from './luma';

const device = CUDADevice.new(1);

Fastify()
    .get('/', (req, reply) => reply.type('text/html').send(createReadStream('index.html')))
    .get('/index.html', (req, reply) => reply.type('text/html').send(createReadStream('index.html')))
    .get('/stream.mp4', async (req, reply) => {
        let loop, outputs;
        try {
            ({ loop, outputs } = await createAnimationLoopInstance(device));
        } catch (err) {
            return reply.type('text/plain').code(500).send(`${err}`);
        }
        reply.type('video/mp4');
        stream.pipeline(outputs
            .withInputFps(window.framerate)
            .withOutputFPS(window.framerate)
            .outputFormat('mp4').outputOptions([
                '-g 1',
                '-c:v libx264',
                '-max_delay 1',
                '-analyzeduration 1000',
                '-fflags nobuffer+flush_packets',
                '-use_wallclock_as_timestamps 1',
                '-preset ultrafast',
                '-tune zerolatency',
                '-movflags frag_keyframe+empty_moov',
                // '-movflags frag_keyframe+empty_moov+rtphint',
            ])
            .on('start', () => loop.start())
            , reply.raw, () => loop.stop());
    })
    .listen(8080);
