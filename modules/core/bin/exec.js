#!/usr/bin/env node

module.exports = function (cmd, args, env = {}) {

  try { require('dotenv').config(); } catch (e) { };

  var Path = require('path');
  var env_ = Object.assign({}, process.env, env);
  var binp = Path.join(__dirname, '../', 'node_modules', '.bin');
  var opts = {
    shell: true,
    stdio: 'inherit',
    cwd: process.cwd(),
    env: Object.assign({}, env_, {
      PATH: `${env_.PATH}:${binp}`
    })
  };

  var name = (() => {
    switch (require('os').platform()) {
      case 'win32':
        return 'windows.sh'
      default:
        return 'linux.sh'
    }
  })();

  return require('child_process').spawnSync(
    Path.join(__dirname, cmd, name), args, opts);
}

if (require.main === module) {
  module.exports(process.argv[2], process.argv.slice(3));
}
