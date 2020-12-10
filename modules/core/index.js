var Path = require('path');

module.exports = {
    cpp_include_path: Path.resolve(__dirname, 'include'),
    ccache_path: Path.resolve(__dirname, '.cache', 'ccache'),
    cpm_source_cache_path: Path.resolve(__dirname, '.cache', 'cpm'),
    cmake_modules_path: Path.resolve(__dirname, 'cmake', 'Modules'),
    loadNativeModule(srcModule, nativeModuleName) {
        let nativeModule, types = ['Release'], errors = [];
        if (process.env.NODE_DEBUG !== undefined || process.env.NODE_ENV === 'debug') {
            types.push('Debug');
        }
        // Adjust `base` path if running in Jest
        let base = Path.dirname(srcModule.id);
        if (Path.basename(base) == 'src') {
            base = Path.join(Path.dirname(base), 'build', 'js');
        }
        base = Path.resolve(base, '..');
        for (let type; type = types.pop();) {
            try {
                if (nativeModule = require(Path.join(base, type, `${nativeModuleName}.node`))) {
                    break;
                }
            } catch (e) { errors.push(e); continue; }
        }
        if (nativeModule) {
            if (typeof nativeModule.init === 'function') {
                return nativeModule.init();
            }
            return nativeModule;
        }
        throw new Error([
            `${nativeModuleName} not found`,
            ...errors.map((e) => e && (e.stack || e.message) || `${e}`)
        ].join('\n'));
    }
};
