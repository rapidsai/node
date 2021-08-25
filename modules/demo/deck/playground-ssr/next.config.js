/** @type {import('next').NextConfig} */
module.exports = {
  reactStrictMode: true,
  generateEtags: false,
  poweredByHeader: false,
  webpack: (config, { buildId, dev, isServer, defaultLoaders, webpack }) => {
    // Note: we provide webpack above so you should not `require` it
    // Perform customizations to webpack config
    // config.plugins.push(new webpack.IgnorePlugin({ resourceRegExp: /.*?\.node/ig }))
    if (isServer) {
      const majorMinor = process.versions.node.split('.').slice(0, 2).join('.');
      config.target = `node${majorMinor}`;
      config.node = {
        global: true,
        __dirname: true,
        __filename: true,
      };
      config.externals.push({
        '@nvidia/cuda': '@nvidia/cuda',
        '@nvidia/glfw': '@nvidia/glfw',
        '@nvidia/webgl': '@nvidia/webgl',
        '@rapidsai/core': '@rapidsai/core',
        '@rapidsai/cudf': '@rapidsai/cudf',
        '@rapidsai/cugraph': '@rapidsai/cugraph',
        '@rapidsai/cuspatial': '@rapidsai/cuspatial',
        '@rapidsai/deck.gl': '@rapidsai/deck.gl',
        '@rapidsai/demo-deck-playground': '@rapidsai/demo-deck-playground',
        '@rapidsai/jsdom': '@rapidsai/jsdom',
        '@rapidsai/rmm': '@rapidsai/rmm',
        'apache-arrow': 'apache-arrow'
      });
    } else {
      // config.plugins.push(new webpack.IgnorePlugin({ resourceRegExp: /.*?\.node/ig }));
      config.plugins.push(new webpack.IgnorePlugin({ resourceRegExp: /.*?child_process/ig }));
      config.resolve.alias['apache-arrow'] = require.resolve('apache-arrow/Arrow.es2015.min.js');
    }
    // console.log(require('util').inspect({ isServer, config }, false, Infinity, true));
    // Important: return the modified config
    return config;
  },
}
