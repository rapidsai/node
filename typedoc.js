module.exports = {
  entryPoints: [
    'modules/cuda/src/index.ts',
    'modules/rmm/src/index.ts',
    'modules/cudf/src/index.ts',
    'modules/cuml/src/index.ts',
    'modules/cugraph/src/index.ts',
    'modules/cuspatial/src/index.ts',
    'modules/io/src/index.ts',
    'modules/deck.gl/src/index.ts',
    'modules/glfw/src/index.ts',
    'modules/jsdom/src/index.ts',
    'modules/webgl/src/index.ts',
    'modules/sql/src/index.ts',
  ],
  out: 'doc',
  name: 'RAPIDS',
  tsconfig: 'tsconfig.json',
  hideGenerator: true,
  excludePrivate: true,
  excludeProtected: true,
  excludeExternals: true,
};
<<<<<<< HEAD
=======

const mergeOptions = (...opts) => Object.assign(
  {}, ...opts, {entryPoints: opts.reduce((es, {entryPoints = []}) => [...es, ...entryPoints], [])});

module.exports = mergeOptions(loadOptions('modules/cuda'),
                              loadOptions('modules/rmm'),
                              loadOptions('modules/cudf'),
                              loadOptions('modules/cuml'),
                              loadOptions('modules/cugraph'),
                              loadOptions('modules/cuspatial'),
                              loadOptions('modules/deck.gl'),
                              loadOptions('modules/glfw'),
                              loadOptions('modules/webgl'),
                              loadOptions('modules/sql'),
                              loadOptions('modules/io'),
                              {
                                out: 'doc',
                                name: 'RAPIDS',
                                tsconfig: 'tsconfig.json',
                                hideGenerator: true,
                                excludePrivate: true,
                                excludeProtected: true,
                                excludeExternals: true,
                              });
>>>>>>> eeaac006 ([FEA] IO Module (#329))
