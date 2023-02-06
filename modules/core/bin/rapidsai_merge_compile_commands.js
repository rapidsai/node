#!/usr/bin/env node

process.exitCode = require('./exec')('merge-compile-commands', process.argv.slice(2)).status;
