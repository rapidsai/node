module.exports = {
    entryPoints: ['src/index.ts'],
    out: 'doc',
    name: '@rapidsai/deck.gl',
    tsconfig: 'tsconfig.json',
    excludePrivate: true,
    excludeProtected: true,
    excludeExternals: true,
};
