const Fastify = require('fastify');
import { createReadStream } from 'fs';
import createAnimationLoopInstance from './luma';

Fastify()
    .get('/', (req, reply) => reply.type('text/html').send(createReadStream('index.html')))
    .get('/index.html', (req, reply) => reply.type('text/html').send(createReadStream('index.html')))
    .get('/mov.mp4', async (req, reply) => {
        createAnimationLoopInstance().then(
            ({ loop, outputs }) => {
                req.raw.once('end', () => loop.stop());
                req.raw.once('destroy', () => loop.stop());
                reply.type('video/mp4').send(outputs);
            },
            (err) => reply.code(500).send(`${err}`)
        );
    })
    .listen(8080);
