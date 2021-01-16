module.exports = {
    entryPoints: ['src'],
    out: 'doc',
    name: '@nvidia/rmm',
    tsconfig: 'tsconfig.json',
    excludePrivate: true,
    excludeProtected: true,
    excludeExternals: true,
};
