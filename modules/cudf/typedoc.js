module.exports = {
    entryPoints: ['src/index.ts'],
    out: 'doc',
    name: '@rapidsai/cudf',
    tsconfig: 'tsconfig.json',
    excludePrivate: true,
    excludeProtected: true,
    excludeExternals: true,
};
