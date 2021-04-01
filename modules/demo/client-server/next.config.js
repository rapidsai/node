module.exports = {
  webpack: (config, { buildId, dev, isServer, defaultLoaders, webpack }) => {
    // Note: we provide webpack above so you should not `require` it
    // Perform customizations to webpack config
    // config.plugins.push(new webpack.IgnorePlugin({ resourceRegExp: /.*?\.node/ig }))
    if (isServer) {
      config.externals.push({
        '@rapidsai/core': '@rapidsai/core',
        '@nvidia/cuda': '@nvidia/cuda',
        '@nvidia/webgl': '@nvidia/webgl',
        '@rapidsai/deck.gl': '@rapidsai/deck.gl',
        '@rapidsai/rmm': '@rapidsai/rmm',
        '@nvidia/glfw': '@nvidia/glfw',
        '@rapidsai/cudf': '@rapidsai/cudf',
        '@rapidsai/cugraph': '@rapidsai/cugraph',
        '@rapidsai/cuspatial': '@rapidsai/cuspatial',
        'apache-arrow': 'apache-arrow'
      });
    }
    // Important: return the modified config
    return config;
  },
}
