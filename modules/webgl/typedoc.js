module.exports = {
    entryPoints: ['src/index.ts'],
    out: 'doc',
    name: '@nvidia/webgl',
    tsconfig: 'tsconfig.json',
    excludePrivate: true,
    excludeProtected: true,
    excludeExternals: true,
};