module.exports = {
  entryPoints: ['src/index.ts'],
  out: 'doc',
  name: '@rapidsai/glfw',
  tsconfig: 'tsconfig.json',
  excludePrivate: true,
  excludeProtected: true,
  excludeExternals: true,
};
