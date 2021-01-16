module.exports = {
    entryPoints: ['src'],
    out: 'doc',
    name: '@nvidia/webgl',
    tsconfig: 'tsconfig.json',
    excludePrivate: true,
    excludeProtected: true,
    excludeExternals: true,
};
