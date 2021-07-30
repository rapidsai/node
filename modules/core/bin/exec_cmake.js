#!/usr/bin/env node

process.exitCode = require('./exec')('cmake-js', process.argv.slice(2)).status;
