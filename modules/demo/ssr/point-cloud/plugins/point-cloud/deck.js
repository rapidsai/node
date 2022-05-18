function makeDeck() {
  const {log: deckLog} = require('@deck.gl/core');
  deckLog.level        = 0;
  deckLog.enable(false);

  const {OrbitView, COORDINATE_SYSTEM, LinearInterpolator} = require('@deck.gl/core');
  const {PointCloudLayer}                                  = require('@deck.gl/layers');
  const {DeckSSR}                                          = require('@rapidsai/deck.gl');
  const {LASLoader}                                        = require('@loaders.gl/las');
  const {registerLoaders}                                  = require('@loaders.gl/core');

  registerLoaders(LASLoader);

  // Data source: kaarta.com
  const LAZ_SAMPLE = 'http://localhost:8080/indoor.0.1.laz';

  const transitionInterpolator = new LinearInterpolator(['rotationOrbit']);

  const makeLayers = (deck) => {
    return [
      new PointCloudLayer({
        id: 'laz-point-cloud-layer',
        data: LAZ_SAMPLE,
        coordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
        getNormal: [0, 1, 0],
        getColor: [255, 255, 255],
        opacity: 0.5,
        pointSize: 0.5
      }),
    ]
  };

  const deck = new DeckSSR({
    createFramebuffer: true,
    initialViewState: {
      target: [0, 0, 0],
      rotationX: 0,
      rotationOrbit: 0,
      orbitAxis: 'Y',
      fov: 50,
      minZoom: 0,
      maxZoom: 10,
      zoom: 1
    },
    layers: makeLayers(null),
    views: [
      new OrbitView({transitionInterpolator}),
    ],
    controller: true,
    parameters: {clearColor: [0.93, 0.86, 0.81, 1]},
    onAfterAnimationFrameRender({_loop}) { _loop.pause(); },
  });

  return {
    deck,
    render() {
      const done = deck.animationLoop.waitForRender();
      deck.setProps({layers: makeLayers(deck)});
      deck.animationLoop.start();
      return done;
    },
  };
}

module.exports = {
  makeDeck: makeDeck,
  openLayerIpcHandles: function() {},
  closeLayerIpcHandles: function() {},
  serializeCustomLayer: function() {}
};
