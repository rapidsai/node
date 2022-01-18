# Simple Graph Demo
A simple graph visualization powered by cuGraph and deck.gl, using a glfw instance. While running, you can select parameters by using the up/down arrow keys or 1-9 and adjust by using left/right arrow keys. Pan and zoom with a mouse.

## Featured Dependencies
@rapidsai/cudf
@rapidsai/cuspatial
@rapidsai/jsdom
@rapidsai/deckgl
@rapidsai/glfw

## Data Requirements
Without passing data, graph demo will default to internal data. Passing graph data assumes a node and edge `.csv` format in `/data` folder.

## Start
Example of starting the graph demo with selected data and preset parameters:
`yarn start
 --width=1920 --height=1080 \
 --nodes=data/netscience.nodes.csv \
 --edges=data/netscience.edges.csv \
 --params='"autoCenter":1,"outboundAttraction":1,"strongGravityMode":1,"jitterTolerance":0.02,"barnesHutTheta":0,"scalingRatio":2,"gravity":0.5,"controlsVisible":0'
 `
