const {DeckGL, GeoJsonLayer} = deck;
const socket                 = io();
var src_layer                = '';
var src_deck                 = '';
var dst_layer                = '';
var dst_deck                 = '';
var start_hour_chart         = '';
var day_chart                = '';

// update all charts in the dashboard, and Dataset size
function updateCharts() {
  updateSrcMap();
  updateDstMap();
  updateDayBar();
  updateHourBar();
  updateDatasetSize();
}

// Reset all charts together, by clearing all filters and calling updateCharts()
function resetAllCharts() {
  query_dict = {};
  // soft clear all selected brushes(with triggering update for a single update command later)
  action = {type: 'brush', areas: []};
  day_chart.dispatchAction(action);
  start_hour_chart.dispatchAction(action);
  updateCharts();
}

// read data on socket connection success, and intiate respective charts
// this is where the app is intialized
socket.on('connect', () => {
  socket.emit('readUberData', () => {
    src_deck         = init_deck('map-src');
    dst_deck         = init_deck('map-dst');
    start_hour_chart = echarts.init(document.getElementById('bar-hour'));
    day_chart        = echarts.init(document.getElementById('bar-day'));
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
  start_hour_chart.on('brushSelected', selectionCallback);
  start_hour_chart.on('brush', resetHourBar);
  day_chart.on('brushSelected', selectionCallback);
  day_chart.on('brush', resetDayBar);
  document.getElementById('reset-src-map').onclick    = resetSrcMap;
  document.getElementById('reset-dst-map').onclick    = resetDstMap;
  document.getElementById('reset-all-charts').onclick = resetAllCharts;
  generateLegend('#l1', {
    0: '0s to 400s',
    500: '400s to 800s',
    900: '800s to 1000s',
    1100: '1000s or higher',
  });
  generateLegend('#l2', {
    0: '0s to 400s',
    500: '400s to 800s',
    900: '800s to 1000s',
    1100: '1000s or higher',
  });
}
