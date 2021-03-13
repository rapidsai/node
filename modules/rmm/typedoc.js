module.exports = {
    entryPoints: ['src/index.ts'],
    out: 'doc',
    name: '@rapidsai/rmm',
    tsconfig: 'tsconfig.json',
    excludePrivate: true,
    excludeProtected: true,
    excludeExternals: true,
};
