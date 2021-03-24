var bar_data_idx = {};  // to maintain the xaxis original values for bar charts( bar_chart object
// only stores modified labels)
var query_dict = {};    // to maintain the queried state of the dashboard

// map container div ids to column names in uber dataframe
const map_columns = {
  'map-src': 'sourceid',
  'map-dst': 'dstid'
};

// update Source Map chart and updateConsole with compute times
async function updateSrcMap() {

  const t0 = performance.now();

  // Normally we'd use something like `query-string` to construct and
  // url-encode the query string from a JSON object, but that's more
  // difficult because we're loading our JS deps via <script/> tags.
  const url = `/uber/tracts/groupby/sourceid/mean?columns=travel_time`;
  const table = await Arrow.Table.from(fetch(url, { method: 'GET' }));

  const ids = table.getChildAt(0);
  const polygons = table.getChildAt(1);
  const rings = polygons.getChildAt(0);
  const coords = rings.getChildAt(0);
  const xys = coords.getChildAt(0);

  const pathIndices = { size: 1, value: rings.data.valueOffsets };
  const polygonIndices = { size: 1, value: polygons.data.valueOffsets };
  const positions = { size: 2, value: xys.data.values };
  const globalFeatureIds = { size: 1, value: ids.data.values };
  const featureIds = { size: 1, value: Uint32Array.from({ length: ids.length }, (_, i) => i) };

  src_deck.setProps({
    layers: [
      get_layer('map-src', {
        points: {
          positions,
          globalFeatureIds,
          featureIds,
          numericProps: {},
          properties: {},
        },
        lines: {
          positions,
          pathIndices,
          globalFeatureIds,
          featureIds,
          numericProps: {},
          properties: {},
        },
        polygons: {
          positions,
          polygonIndices,
          primitivePolygonIndices: pathIndices,
          globalFeatureIds,
          featureIds,
          numericProps: {},
          properties: {},
        }
      })
    ]
  });

  updateConsole(`groupby('sourceid').mean().query(${JSON.stringify(query_dict)
    })<br> <b>Time taken: </b> ${(performance.now() - t0).toFixed(2)}ms`);
}

// update Destination Map chart and updateConsole with compute times
async function updateDstMap() {
  var t0 = new Date().getTime();
  socket.emit('groupby',
    'dstid',
    'mean',
    { type: 'geojson', properties: ['travel_time'], geojsonProp: 'MOVEMENT_ID' },
    query_dict,
    (data) => {
      dst_deck.setProps({ layers: [get_layer('map-dst', data)] });
      var t1 = new Date().getTime();
      updateConsole('groupby(`dstid`).mean() .query(' + JSON.stringify(query_dict) +
        ')<br> <b>Time taken: </b>' + (t1 - t0) + 'ms');
    });
}

// update Day Bar chart and updateConsole with compute times
async function updateDayBar() {
  var t0 = new Date().getTime();
  socket.emit(
    'groupby', 'day', 'count', { type: 'arrow', column: 'sourceid' }, query_dict, (data) => {
      option = generateOptions('day', data);
      bar_data_idx['day'] = data.reduce((a, c) => { return [...a, c['day']] }, []),
        option.xAxis.data = Array.from({ length: 31 }, (v, k) => k + 1);
      day_chart.setOption(option);
      var t1 = new Date().getTime();
      updateConsole('groupby(`day`).count() .query(' + JSON.stringify(query_dict) +
        ')<br> <b>Time taken: </b>' + (t1 - t0) + 'ms');
    })
}

// update Start Hour Bar chart and updateConsole with compute times
async function updateHourBar() {
  var t0 = new Date().getTime();
  socket.emit(
    'groupby', 'start_hour', 'count', { type: 'arrow', column: 'sourceid' }, query_dict, (data) => {
      option = generateOptions('start_hour', data);
      bar_data_idx['start_hour'] = data.reduce((a, c) => { return [...a, c['start_hour']] }, []),
        option.xAxis.type = 'category';
      option.xAxis.data = ['AM Peak', 'Midday', 'PM Peak', 'Evening', 'Early Morning'];
      option.xAxis.axisLabel.rotate = 45;
      start_hour_chart.setOption(option);
      var t1 = new Date().getTime();
      updateConsole('groupby(`start_hour`).count() .query(' + JSON.stringify(query_dict) +
        ')<br> <b>Time taken: </b>' + (t1 - t0) + 'ms');
    })
}

// update Dataset size
async function updateDatasetSize() {
  socket.emit(
    'get-dataset-size',
    query_dict,
    (size) => { document.getElementById('data-points').innerHTML = d3.format(',')(data_points); });
}

// reset sourceMap filters and update all charts
function resetSrcMap(e) {
  console.log(e);
  delete query_dict.sourceid;
  updateCharts();
}

// reset dstMap filters and update all charts
function resetDstMap() {
  delete query_dict.dstid;
  updateCharts();
}

// reset hourBar filters and update all charts
function resetHourBar(params) {
  if (params == undefined || params.command == 'clear') {
    delete query_dict.start_hour;
    updateCharts();
  }
}

// reset dayBar filters and update all charts
function resetDayBar(params) {
  if (params == undefined || params.command == 'clear') {
    delete query_dict.day;
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
  d3.scaleThreshold().domain([0, 400, 800, 1000, 2000, 4000]).range(COLOR_SCALE);

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
    series:
      [{ type: 'bar', id: column, data: data.reduce((a, c) => { return [...a, c.sourceid] }, []) }],
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
  };
}

// colorScale function passed to deckGL.GeoJSONLayer.getFillColor
function colorScale(x, id, containerID) {
  if (x !== undefined && id !== undefined) {
    if (map_columns[containerID] in query_dict) {
      if (query_dict[map_columns[containerID]] !== parseInt(id)) {
        return ([255, 255, 255, 10]);
      } else {
        return thresholdScale(x);
      }
    } else {
      return thresholdScale(x);
    }
  }
}

// passed to deckGL.GeoJSONLayer.getElevation
function filterByValue(id, containerID) {
  if (id !== undefined) {
    query_dict[map_columns[containerID]] = parseInt(id);
    updateCharts();
  }
}

function get_layer(containerID, data) {
  return new GeoJsonLayer({
    data: data,
    highlightColor: [200, 200, 200, 200],
    autoHighlight: true,
    wireframe: false,
    pickable: true,
    stroked: false,  // only on extrude false
    filled: true,
    extruded: true,
    lineWidthScale: 10,  // only if extrude false
    lineWidthMinPixels: 1,
    getRadius: 100,  // only if points
    getLineWidth: 10,
    opacity: 50,
    getElevation: f => f?.properties?.travel_time * 10,
    getFillColor: f => colorScale(f?.properties?.travel_time, f?.properties?.MOVEMENT_ID, containerID),
    onClick: f => filterByValue(f.object?.properties?.MOVEMENT_ID, containerID),
    getLineColor: [0, 188, 212, 100],
  });
}

// passed to DeckGL.tooltip
function getTooltip({ object }) {
  return object && `${object.properties.DISPLAY_NAME}
travel_time: ${object.properties.travel_time}
MOVEMENT_ID: ${object.properties.MOVEMENT_ID}
`;
}

// initialize DeckGL object given a containerID
function init_deck(containerID) {
  return new DeckGL({
    mapStyle: 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json',
    initialViewState: { longitude: -122, latitude: 37, zoom: 6, maxZoom: 16, pitch: 0, bearing: 0 },
    controller: true,
    getTooltip,
    container: containerID
  });
}
