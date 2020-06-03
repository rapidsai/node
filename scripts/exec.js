#!/usr/bin/env node

var name = (() => {
    switch (require('os').platform()) {
        case 'win32':
            return 'win32.sh'
        default:
            return 'linux.sh'
    }
})();

var Path = require('path');
var root = Path.join(__dirname, '../');
var path = Path.join(__dirname, process.argv[2]);
var pcwd = Path.join(path, Path.relative(path, root));
var proc = require('child_process').spawn(
    Path.join(path, name),
    process.argv.slice(3),
    { stdio: 'inherit', cwd: pcwd }
);

['SIGTERM', 'SIGINT', 'SIGBREAK', 'SIGHUP'].forEach((signal) => {
    process.on(signal, () => proc.kill(signal))
});

proc.on('exit', (code, signal) => {
    // exit code could be null when OS kills the process(out of memory, etc) or
    // due to node handling it but if the signal is SIGINT the user exited the
    // process so we want exit code 0
    process.exit(code === null ? signal === 'SIGINT' ? 0 : 1 : code);
});
