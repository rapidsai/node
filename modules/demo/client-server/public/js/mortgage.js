
const {DeckGL, GeoJsonLayer} = deck;
const socket                 = io();

const COLOR_SCALE = [
  [49, 130, 189, 100],
  [107, 174, 214, 100],
  [123, 142, 216, 100],
  [226, 103, 152, 100],
  [255, 0, 104, 100],
  [50, 50, 50, 100]
];

var geojsonLayer = '';
var deck         = '';

function calcZip3BinColors(value) {
  if (value == undefined) {
    return (COLOR_SCALE[5])

  } else if (value <= 0.196) {
    return (COLOR_SCALE[0])

  } else if (value > 0.196 && value <= 0.198) {
    return (COLOR_SCALE[1])

  } else if (value > 0.198 && value <= 0.200) {
    return (COLOR_SCALE[2])

  } else if (value > 0.200 && value <= 0.202) {
    return (COLOR_SCALE[3])

  } else if (value > 0.202) {
    // red alert
    return (COLOR_SCALE[4])

  } else {
    return (COLOR_SCALE[5])
  }
}

function init_deck() {
  geojsonLayer = new GeoJsonLayer({
    highlightColor: [200, 200, 200, 200],
    autoHighlight: true,
    wireframe: true,
    pickable: true,
    stroked: false,  // only on extrude false
    filled: true,
    extruded: true,
    lineWidthScale: 10,  // only if extrude false
    lineWidthMinPixels: 1,
    getRadius: 100,  // only if points
    getLineWidth: 10,
    opacity: 50,
    getElevation: f => f.properties.current_actual_upb,
    getFillColor: f => calcZip3BinColors(f.properties.delinquency_12_prediction),
    getLineColor: [0, 188, 212, 100],
  });
  deck         = new DeckGL({
    mapStyle: 'https://basemaps.cartocdn.com/gl/positron-nolabels-gl-style/style.json',
    initialViewState: {longitude: -101, latitude: 37, zoom: 3, maxZoom: 16, pitch: 0, bearing: 0},
    controller: true,
    layers: [geojsonLayer],
    getTooltip
  });
}

socket.on('connect', () => {
  socket.emit('readMortgageData',
              () => {socket.emit(
                'groupby',
                'zip',
                'mean',
                ['dti', 'delinquency_12_prediction', 'borrower_credit_score', 'current_actual_upb'],
                'ZIP3',
                (data) => {
                  init_deck();
                  geojsonLayer.props.data = data;
                })});
});

function getTooltip({object}) {
  return object && `current_actual_upb: ${object.properties.current_actual_upb}
        delinquency_12_prediction: ${object.properties.delinquency_12_prediction}`;
}
