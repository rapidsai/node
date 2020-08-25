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

        float curvature = 2.5;
        float stroke = strokeWidth * curvature;
        float eindex = float(bundle.x) + mod(float(bundle.x), 2.);
        float bcount = float(bundle.y);
        float direction = mix(1., -1., mod(bcount, 2.));

        // If all the edges in the bundle fit into MAX_BUNDLE_SIZE,
        // separate the edges without overlap via 'stroke * eindex'.
        // Otherwise allow edges to overlap.
        float size = mix(
            stroke * eindex * curvature,
            (MAX_BUNDLE_SIZE * .5 / stroke) * (eindex / bcount),
            step(MAX_BUNDLE_SIZE, bcount * stroke)
        );

        vec3 diff = normalize(targetPositions - sourcePositions);
        vec3 midp = (targetPositions + sourcePositions) * .5;
        vec3 unit = vec3(-diff.y, diff.x, 1.);

        controlPoints = vec3((midp + (unit * size * direction)).xy, 0.);
    }
}
`;
