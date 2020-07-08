export default `\
#version 410

#define MAX_BUNDLE_SIZE 100.0

in uvec2 edge;
in uvec2 bundle;

uniform sampler2D nodeXPositions;
uniform sampler2D nodeYPositions;
uniform uint loadedNodeCount;
uniform float strokeWidth;
uniform uint width;

out vec3 controlPoints;
out vec3 sourcePositions;
out vec3 targetPositions;

ivec2 getTexCoord(uint id) {
    uint y = id / width;
    uint x = id - width * y;
    return ivec2(x, y);
}

void main(void) {

    controlPoints = vec3(0., 0., 1.);
    sourcePositions = vec3(0., 0., 1.);
    targetPositions = vec3(0., 0., 1.);

    if (edge.x < loadedNodeCount) {
        ivec2 xIdx = getTexCoord(edge.x);
        sourcePositions = vec3(
            texelFetch(nodeXPositions, xIdx, 0).x,
            texelFetch(nodeYPositions, xIdx, 0).x,
            0.
        );
    }
    if (edge.y < loadedNodeCount) {
        ivec2 yIdx = getTexCoord(edge.y);
        targetPositions = vec3(
            texelFetch(nodeXPositions, yIdx, 0).x,
            texelFetch(nodeYPositions, yIdx, 0).x,
            0.
        );
    }

    // Compute the quadratic bezier control point for this edge
    if ((sourcePositions.z + targetPositions.z) < 1.0) {

        uint uindex = bundle.x;
        float bundleSize = bundle.y;
        float stroke = max(strokeWidth, 1.);
        float eindex = uindex + mod(uindex, 2);
        // int direction = int(mix(1, -1, mod(uindex, 2)));
        int direction = int(mix(1, -1, mod(bundleSize, 2)));

        // If all the edges in the bundle fit into MAX_BUNDLE_SIZE,
        // separate the edges without overlap via 'stroke * eindex'.
        // Otherwise allow edges to overlap.
        float size = mix(
            stroke * eindex,
            (MAX_BUNDLE_SIZE * .5 / stroke)
                * (eindex / bundleSize),
            step(MAX_BUNDLE_SIZE, bundleSize * strokeWidth)
        ) + 15.0;

        vec3 midp = (sourcePositions + targetPositions) * 0.5;
        vec3 dist = (targetPositions - sourcePositions);
        vec3 unit = normalize(dist) * vec3(-1., 1., 1.);

        // handle horizontal and vertical edges
        if (unit.x == 0. || unit.y == 0.) {
            unit = mix(
                vec3(1., 0., unit.z), // if x is 0, x=1, y=0
                vec3(0., 1., unit.z), // if y is 0, x=0, y=1
                step(0.5, unit.x)
            );
        }

        controlPoints = vec3((midp + (unit * size * direction)).xy, 0.);
    }
}
`;
