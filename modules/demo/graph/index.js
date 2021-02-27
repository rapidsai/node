// Copyright (c) 2020, NVIDIA CORPORATION.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

require('segfault-handler').registerHandler('./crash.log');

require('@babel/register')({
    cache: false,
    babelrc: false,
    presets: [
        ["@babel/preset-env", { "targets": { "node": "current" }}],
        ['@babel/preset-react', { "useBuiltIns": true }]
    ]
});

// Change cwd to the example dir so relative file paths are resolved
process.chdir(__dirname);

const url = process.argv.slice(2).find((arg) => arg.includes('tcp://'));
const serve = process.argv.slice(2).some((arg) => arg.includes('--serve'));
const nodes = process.argv.slice(2).find((arg) => arg.includes('--nodes='));
const edges = process.argv.slice(2).find((arg) => arg.includes('--edges='));

if (!serve) {
    module.exports = require('@nvidia/glfw').createReactWindow(`${__dirname}/src/index.js`, true);
}

if (require.main === module) {
    if (serve) {
        require(`./server.js`)({
            url: url ? require('url').parse(url) : undefined,
            nodes: nodes ? nodes.slice('--nodes='.length) : undefined,
            edges: edges ? edges.slice('--edges='.length) : undefined,
        });
    } else {
        module.exports.open({
            visible: true,
            transparent: false,
            _title: 'graph client',
            url: url ? require('url').parse(url) : undefined,
            nodes: nodes ? nodes.slice('--nodes='.length) : undefined,
            edges: edges ? edges.slice('--edges='.length) : undefined,
        });
    }
}
