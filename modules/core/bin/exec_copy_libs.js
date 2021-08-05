#!/usr/bin/env node

process.exitCode = require('./exec')('copy-libs', process.argv.slice(2)).status;
