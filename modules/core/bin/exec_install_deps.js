#!/usr/bin/env node

process.exitCode = require('./exec')('install-deps', process.argv.slice(2)).status;
