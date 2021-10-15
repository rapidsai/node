module.exports = {
  eslint: {
    ignoreDuringBuilds: true,
  },
  webpack: (config, {buildId, dev, isServer, defaultLoaders, webpack}) => {
    // Note: we provide webpack above so you should not `require` it
    // Perform customizations to webpack config
    // config.plugins.push(new webpack.IgnorePlugin({ resourceRegExp: /.*?\.node/ig }))
    if (isServer) {
      config.externals.push({
        '@rapidsai/core': '@rapidsai/core',
        '@rapidsai/cuda': '@rapidsai/cuda',
        '@rapidsai/webgl': '@rapidsai/webgl',
        '@rapidsai/deck.gl': '@rapidsai/deck.gl',
        '@rapidsai/rmm': '@rapidsai/rmm',
        '@rapidsai/glfw': '@rapidsai/glfw',
        '@rapidsai/cudf': '@rapidsai/cudf',
        '@rapidsai/cugraph': '@rapidsai/cugraph',
        '@rapidsai/cuspatial': '@rapidsai/cuspatial',
        'apache-arrow': 'apache-arrow'
      });
    } else {
      config.resolve.alias['apache-arrow'] = require.resolve('apache-arrow/Arrow.es2015.min.js');
    }
    // console.log(require('util').inspect({ isServer, config }, false, Infinity, true));
    // Important: return the modified config
    return config;
  },
}
