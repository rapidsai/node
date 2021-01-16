module.exports = {
    entryPoints: ['src'],
    out: 'doc',
    name: '@nvidia/deck.gl',
    tsconfig: 'tsconfig.json',
    excludePrivate: true,
    excludeProtected: true,
    excludeExternals: true,
};
