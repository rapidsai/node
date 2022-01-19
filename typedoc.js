module.exports = {
  entryPoints: [
    'modules/cuda/src/index.ts',
    'modules/rmm/src/index.ts',
    'modules/cudf/src/index.ts',
    'modules/cuml/src/index.ts',
    'modules/cugraph/src/index.ts',
    'modules/cuspatial/src/index.ts',
    'modules/deck.gl/src/index.ts',
    'modules/glfw/src/index.ts',
    'modules/webgl/src/index.ts',
    'modules/sql/src/index.ts',
    'modules/io/src/index.ts'
  ],
  out: 'doc',
  name: 'RAPIDS',
  tsconfig: 'tsconfig.json',
  hideGenerator: true,
  excludePrivate: true,
  excludeProtected: true,
  excludeExternals: true,
};
