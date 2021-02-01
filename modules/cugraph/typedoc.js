module.exports = {
    entryPoints: ['src/index.ts'],
    out: 'doc',
    name: '@nvidia/cugraph',
    tsconfig: 'tsconfig.json',
    excludePrivate: true,
    excludeProtected: true,
    excludeExternals: true,
};
