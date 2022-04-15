"""
A Story: Adding node-rapids to sigma.js
"""

Get node-rapids
  github.com/rapidsai/node-rapids

Configure node-rapids to use SCCACHE:
  echo 'CUDAARCHS=ALL' | tee -a .env

Build node-rapids
  yarn nuke:from:orbit

Enter the node-rapids container via shell
  yarn run:docker:devel

Get sigma.js
  github.com/jacomyal/sigma.js

Load a cudf dataframe in
  sigma.js/demo/src/views/Root.tsx`

Create a cudf dataframe using the "nodes" object structure in
 `/public/data/dataset.json`

Remove local node_modules caching so we can edit sigma.js as we explore its data format
  demo/config/webpack.config.js
    cacheDirectory:false

  NodeProgram.prototype.process = function(data, hidden, offset) copies the x/y coordinates and
    size and color attributes into the Float32Array buffer to move to GPU. 

  program.bufferData()` copies``this.array` to the `gl.ARRAY_BUFFER`

Ask: Why use runMiddleware?

TODO:Improve toString and other StringSeries functions

TODO:Improve Series logging and _col representation

Is the uber/mortgage api requests actually putting client data in the response?

