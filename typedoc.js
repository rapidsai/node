const Path = require('path');
const loadOptions = (base) => {
    const opts = require(Path.resolve(base, 'typedoc.js'));
    opts.entryPoints = opts.entryPoints.map((entryPoint) => Path.resolve(base, entryPoint));
    return opts;
};

const mergeOptions = (...opts) => Object.assign({}, ...opts, {
    entryPoints: opts.reduce((es, { entryPoints = [] }) => [...es, ...entryPoints], [])
});

module.exports = mergeOptions(
    loadOptions('modules/cuda'),
    loadOptions('modules/rmm'),
    loadOptions('modules/cudf'),
    loadOptions('modules/cugraph'),
    loadOptions('modules/cuspatial'),
    loadOptions('modules/deck.gl'),
    loadOptions('modules/glfw'),
    loadOptions('modules/webgl'),
    loadOptions('modules/blazingsql'),
    {
        out: 'doc',
        name: 'RAPIDS',
        tsconfig: 'tsconfig.json',
        hideGenerator: true,
        excludePrivate: true,
        excludeProtected: true,
        excludeExternals: true,
    }
);
