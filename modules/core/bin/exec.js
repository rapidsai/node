#!/usr/bin/env node

var Path = require('path');
var binp = Path.join(__dirname, '../', 'node_modules', '.bin');
var opts = {
    shell: true,
    stdio: 'inherit',
    cwd: process.cwd(),
    env: Object.assign({}, process.env, {
        PATH: `${process.env.PATH}:${binp}` })
};

module.exports = function(cmd, args) {

    var name = (() => {
        switch (require('os').platform()) {
            case 'win32':
                return 'windows.sh'
            default:
                return 'linux.sh'
        }
    })();

    var proc = require('child_process').spawn(
        Path.join(__dirname, cmd, name), args, opts);

    ['SIGTERM', 'SIGINT', 'SIGBREAK', 'SIGHUP'].forEach((signal) => {
        process.on(signal, () => proc.kill(signal))
    });

    proc.on('exit', (code, signal) => {
        // exit code could be null when OS kills the process(out of memory, etc) or
        // due to node handling it but if the signal is SIGINT the user exited the
        // process so we want exit code 0
        process.exit(code === null ? signal === 'SIGINT' ? 0 : 1 : code);
    });
}

if (require.main === module) {
    module.exports(process.argv[2], process.argv.slice(3));
}
