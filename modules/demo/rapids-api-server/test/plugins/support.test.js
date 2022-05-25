'use strict'

const {test}  = require('tap')
const Fastify = require('fastify')
const Support = require('../../plugins/support')

describe('gpu_cache tests',
         async (t) => {
           test('support works standalone', async (t) => { console.log('Gpu cache caches'); })});

// You can also use plugin with opts in fastify v2
//
// test('support works standalone', (t) => {
//   t.plan(2)
//   const fastify = Fastify()
//   fastify.register(Support)
//
//   fastify.ready((err) => {
//     t.error(err)
//     t.equal(fastify.someSupport(), 'hugs')
//   })
// })
