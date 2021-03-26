const { DeckGL, GeoJsonLayer } = deck;
const socket = io();
let layer = '';
let deck_chart = '';
// let risk_chart = '';
let dti_chart = '';
let credit_chart = '';

// update all charts in the dashboard, and Dataset size
function updateCharts() {
  updateMap();
  // updateRiskBar();
  updateDtiBar();
  updateCreditBar();
  updateDatasetSize();
}

// Reset all charts together, by clearing all filters and calling updateCharts()
function resetAllCharts() {
  query_dict = {};
  // soft clear all selected brushes(with triggering update for a single update command later)
  action = { type: 'brush', areas: [] };
  // risk_chart.dispatchAction(action);
  dti_chart.dispatchAction(action);
  credit_chart.dispatchAction(action);
  updateCharts();
}

// read data on socket connection success, and intiate respective charts
// this is where the app is intialized
socket.on('connect', () => {
  socket.emit('readMortgageData', () => {
    deck_chart = init_deck('map');
    // risk_chart = echarts.init(document.getElementById('bar-risk'));
    dti_chart = echarts.init(document.getElementById('bar-dti'));
    credit_chart = echarts.init(document.getElementById('bar-credit'));
    initEventListeners();
    updateCharts();
  });
});

// update data-points div whenever `data-points-update` event is received from socket.io server
socket.on('data-points-update', (data_points) => {
  document.getElementById('data-points').innerHTML = d3.format(',')(data_points);
})

// initialize event listeners and legends after the data has been loaded and charts have been
// initialized
function initEventListeners() {
  // risk_chart.on('brushSelected', selectionCallback);
  // risk_chart.on('brush', resetRiskBar);
  dti_chart.on('brushSelected', selectionCallback);
  dti_chart.on('brush', resetDtiBar);
  credit_chart.on('brushSelected', selectionCallback);
  credit_chart.on('brush', resetCreditBar);
  document.getElementById('reset-map').onclick = resetMap;

  document.getElementById('reset-all-charts').onclick = resetAllCharts;
  generateLegend('#legend-map', {
    "0.100": '0 to 0.196',
    "0.197": '0.196 to 0.198',
    "0.199": '0.198 to 0.200',
    "0.201": '0.200 to 0.202',
    "0.203": '0.202 to 0.2+',
  });
}
