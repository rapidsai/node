module.exports = {
  future: { webpack5: true },
  webpack: (config, { buildId, dev, isServer, defaultLoaders, webpack }) => {
    // Note: we provide webpack above so you should not `require` it
    // Perform customizations to webpack config
    // config.plugins.push(new webpack.IgnorePlugin({ resourceRegExp: /.*?\.node/ig }))
    if (isServer) {
      config.module.rules.push({
        test: /\\.node$/, loader: 'node-loader'
      });
    }
    // Important: return the modified config
    return config
  },
}
