module.exports = {
    entryPoints: ['src'],
    out: 'doc',
    name: '@nvidia/cuda',
    tsconfig: 'tsconfig.json',
    excludePrivate: true,
    excludeProtected: true,
    excludeExternals: true,
};
