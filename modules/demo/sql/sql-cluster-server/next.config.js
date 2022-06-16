module.exports = {
  eslint: {
    ignoreDuringBuilds: true,
  },
  webpack: (config, {buildId, dev, isServer, defaultLoaders, webpack}) => {
    if (!isServer) {
      config.resolve.alias['apache-arrow'] = require.resolve('apache-arrow/Arrow.es2015.min');
    }
    return config;
  },
}
