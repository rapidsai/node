var Path = require('path');

module.exports = {
    cpp_include_path: Path.resolve(__dirname, 'include'),
    ccache_path: Path.resolve(__dirname, '.cache', 'ccache'),
    cpm_source_cache_path: Path.resolve(__dirname, '.cache', 'cpm'),
    cmake_modules_path: Path.resolve(__dirname, 'cmake', 'Modules'),
};
