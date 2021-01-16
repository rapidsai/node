module.exports = {
    entryPoints: ['src'],
    out: 'doc',
    name: '@nvidia/cugraph',
    tsconfig: 'tsconfig.json',
    excludePrivate: true,
    excludeProtected: true,
    excludeExternals: true,
};
