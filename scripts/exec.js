#!/usr/bin/env node

try {
  require('dotenv').config();
} catch (e) { }

var name = (() => {
  switch (require('os').platform()) {
    case 'win32': return 'win32.sh';
    default: return 'linux.sh';
  }
})();

var Path = require('path');
var rootdir = Path.join(__dirname, '../');
var cmdpath = Path.join(__dirname, process.argv[2]);
var cwd = Path.join(cmdpath, Path.relative(cmdpath, rootdir));

process.exitCode = require('child_process').spawnSync(
  Path.join(cmdpath, name),
  process.argv.slice(3),
  { stdio: 'inherit', cwd }
).status;
