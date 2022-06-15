'use strict'

const {test}  = require('tap')
const {build} = require('../helper')

test('root returns API description', async (t) => {
  const app = await build(t)
  const res = await app.inject({url: '/'})
  t.same(JSON.parse(res.payload), {
    'graphology': {
      'description': 'The graphology api provides GPU acceleration of graphology datasets.',
      'schema': {
        'read_json': {
          'filename': 'A URI to a graphology json dataset file.',
          'returns': 'Result OK/Not Found/Fail'
        },
        'list_tables':
          {'returns': 'An object containing graphology related datasets resident on GPU memory.'},
        ':table': {
          ':column': 'The name of the column you want to request.',
          'returns': 'An arrow buffer of the column contents.'
        }
      }
    }
  });
})
