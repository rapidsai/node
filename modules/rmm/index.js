// process.env.LD_LIBRARY_PATH = [
//     process.env.LD_LIBRARY_PATH,
//     require('path').join(__dirname, 'build', 'lib')
// ].filter(Boolean).join(':')

module.exports = require('./build/js/index');
