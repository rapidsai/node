var bar_data_idx = {};  // to maintain the xaxis original values for bar charts( bar_chart object
// only stores modified labels)
var query_dict = {};    // to maintain the queried state of the dashboard


// update Source Map chart and updateConsole with compute times
function updateMap() {
  var t0 = new Date().getTime();
  socket.emit(
    'groupby',
    'zip',
    'mean',
    { type: 'geojson', properties: ['dti', 'delinquency_12_prediction', 'current_actual_upb', 'borrower_credit_score'], geojsonProp: 'ZIP3' },
    query_dict,
    (data) => {
      deck_chart.setProps({ layers: [get_layer(data)] });
      var t1 = new Date().getTime();
      updateConsole('groupby(`zip`).mean() .query(' + JSON.stringify(query_dict) +
        ')<br> <b>Time taken: </b>' + (t1 - t0) + 'ms');
    })
}

// update Start Hour Bar chart and updateConsole with compute times
function updateDtiBar() {
  var t0 = new Date().getTime();
  socket.emit(
    'groupby', 'dti', 'count', { type: 'arrow', column: 'zip' }, query_dict, (data) => {
      option = generateOptions('dti', data);
      bar_data_idx['dti'] = data.reduce((a, c) => { return [...a, c['dti']] }, []);
      dti_chart.setOption(option);
      var t1 = new Date().getTime();
      updateConsole('groupby(`dti`).count() .query(' + JSON.stringify(query_dict) +
        ')<br> <b>Time taken: </b>' + (t1 - t0) + 'ms');
    })
}

// update Start Hour Bar chart and updateConsole with compute times
function updateCreditBar() {
  var t0 = new Date().getTime();
  socket.emit(
    'groupby', 'borrower_credit_score', 'count', { type: 'arrow', column: 'zip' }, query_dict, (data) => {
      option = generateOptions('borrower_credit_score', data);
      bar_data_idx['borrower_credit_score'] = data.reduce((a, c) => { return [...a, c['borrower_credit_score']] }, []);
      credit_chart.setOption(option);
      var t1 = new Date().getTime();
      updateConsole('groupby(`borrower_credit_score`).count() .query(' + JSON.stringify(query_dict) +
        ')<br> <b>Time taken: </b>' + (t1 - t0) + 'ms');
    })
}

// update Dataset size
function updateDatasetSize() {
  socket.emit(
    'get-dataset-size',
    query_dict,
    (size) => { document.getElementById('data-points').innerHTML = d3.format(',')(data_points); });
}

// reset sourceMap filters and update all charts
function resetMap(e) {
  delete query_dict.zip;
  updateCharts();
}

// reset dayBar filters and update all charts
function resetDtiBar(params) {
  if (params == undefined || params.command == 'clear') {
    delete query_dict.dti;
    updateCharts();
  }
}


// reset dayBar filters and update all charts
function resetCreditBar(params) {
  if (params == undefined || params.command == 'clear') {
    delete query_dict.borrower_credit_score;
    updateCharts();
  }
}

// function called on brush select event for bar charts
function selectionCallback(params) {
  if (params.batch.length > 0 && params.batch[0].areas.length > 0) {
    seriesID = params.batch[0].selected[0].seriesId;
    query_dict[seriesID] = params.batch[0].areas[0].coordRange.map(r => {
      var idx = (parseInt(r) >= 0) ? parseInt(r) : 0;
      idx = (idx < bar_data_idx[seriesID].length) ? idx : bar_data_idx[seriesID].length - 1;
      return bar_data_idx[seriesID][idx];
    });
    updateCharts();
  }
}

// color scale for choropleth charts
const COLOR_SCALE = [
  [49, 130, 189, 100],
  [107, 174, 214, 100],
  [123, 142, 216, 100],
  [226, 103, 152, 100],
  [255, 0, 104, 100],
];
const thresholdScale =
  d3.scaleThreshold().domain([0, 0.196, 0.198, 0.200, 0.202]).range(COLOR_SCALE);

// generate default options for echart bar charts
function generateOptions(column, data) {
  return {
    xAxis: {
      type: 'category',
      axisLabel: { color: 'white' },
      splitLine: { show: false },
      name: 'Trips per ' + column,
      nameLocation: 'middle',
      nameGap: 50,
      datasetIndex: 0
    },
    brush: {
      id: column,
      toolbox: ['lineX', 'clear'],
      throttleType: 'debounce',
      throttleDelay: 300,
      xAxisIndex: Object.keys(data)
    },
    yAxis: {
      type: 'value',
      axisLabel: { formatter: d3.format('.2s'), color: 'white' },
      splitLine: { show: false },
    },
    dataset: {
      source: data
    },
    series:
      [{ type: 'bar', id: column }],//, data: data.reduce((a, c) => { return [...a, c.zip] }, []) }],
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
  };
}

// colorScale function passed to deckGL.GeoJSONLayer.getFillColor
function colorScale(x, id) {
  if ('zip' in query_dict) {
    if (query_dict['zip'] !== parseInt(id)) {
      return ([255, 255, 255, 10]);
    } else {
      return thresholdScale(x);
    }
  } else {
    return thresholdScale(x);
  }
}

// passed to deckGL.GeoJSONLayer.getElevation
function filterByValue(id) {
  query_dict['zip'] = parseInt(id);
  updateCharts();
}

function get_layer(data) {
  return new GeoJsonLayer({
    data: data,
    highlightColor: [200, 200, 200, 200],
    autoHighlight: true,
    wireframe: false,
    pickable: true,
    stroked: true,  // only on extrude false
    filled: true,
    extruded: true,
    lineWidthScale: 10,  // only if extrude false
    lineWidthMinPixels: 1,
    getRadius: 100,  // only if points
    getLineWidth: 10,
    opacity: 50,
    getElevation: f => f.properties.current_actual_upb,
    getFillColor: f => colorScale(f.properties.delinquency_12_prediction, f.properties.ZIP3),
    getLineColor: [0, 188, 212, 100],
    onClick: f => filterByValue(f.object.properties.ZIP3),
  });

}

// passed to DeckGL.tooltip
function getTooltip({ object }) {
  return object && `dti: ${object.properties.dti}
        zip: ${object.properties.ZIP3}
        borrower_credit_score: ${object.properties.borrower_credit_score}
        current_actual_upb: ${object.properties.current_actual_upb}
        delinquency_12_prediction: ${object.properties.delinquency_12_prediction}`;
}

// initialize DeckGL object given a containerID
function init_deck(containerID) {
  return new DeckGL({
    mapStyle: 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json',
    initialViewState: { longitude: -101, latitude: 37, zoom: 3, maxZoom: 16, pitch: 0, bearing: 0 },
    controller: true,
    getTooltip,
    container: containerID
  });

}
