const express = require('express');
const { DataFrame } = require('@rapidsai/cudf');
const path = require('path');
const router = express.Router();
const fetch = require('node-fetch');
const { performance } = require('perf_hooks');
const settings = {
  method: 'Get'
};

let gjson = '';
let df = undefined;
let datasetLoaded = '';

// read Mortgage Dataset from disk using cudf.DataFrame.readCSV, and fetch geoJSON file
function readMortgageData(callback) {
  var t0 = performance.now();
  console.log('readMortgageData');
  df = DataFrame.readCSV({
    header: 0,
    sourceType: 'files',
    sources: [path.join(__dirname, '../public/data/m.csv')],
    dataTypes: {
      index: 'int16',
      zip: 'int32',
      dti: 'float32',
      current_actual_upb: 'float32',
      borrower_credit_score: 'int16',
      load_id: 'int32',
      delinquency_12_prediction: 'float32',
      seller_name: 'int16'
    }
  });
  datasetLoaded = 'mortgage';
  let url =
    'https://raw.githubusercontent.com/rapidsai/cuxfilter/GTC-2018-mortgage-visualization/javascript/demos/GTC%20demo/src/data/zip3-ms-rhs-lessprops.json';

  fetch(url, settings).then(res => res.json()).then((json) => {
    gjson = json;
    console.log('readMortgageData complete');
    var t1 = performance.now();
    console.log('Time Taken:', (t1 - t0).toFixed(2), 'ms');
    callback(true);
  });
}

// read Uber Dataset from disk using cudf.DataFrame.readCSV, and fetch geoJSON file
function readUberData(callback) {
  var t0 = performance.now();
  console.log('readUberData');
  df = DataFrame.readCSV({
    header: 0,
    sourceType: 'files',
    // sources: [path.join(__dirname, '../public/data/san_fran_uber.csv')],
    sources: [('public/data/san_fran_uber.csv')],
    dataTypes: {
      sourceid: 'int16',
      dstid: 'int16',
      month: 'int8',
      day: 'int8',
      start_hour: 'int8',
      end_hour: 'int8',
      travel_time: 'float32'
    }
  });
  datasetLoaded = 'uber';

  let url = 'https://movement.uber.com/travel-times/1_censustracts.json'
  fetch(url, settings).then(res => res.json()).then((json) => {
    gjson = json;
    console.log('readUberData complete');
    var t1 = performance.now();
    console.log('Time Taken:', (t1 - t0).toFixed(2), 'ms');
    callback(true);
  });
}

// release data from GPU memory
function deleteData() {
  delete df;
  datasetLoaded = '';
}

/**
 * transform Arrow.Table to the format
 * [index[0]: {property: propertyValue, ....},
 * index[1]: {property: propertyValue, ....},
 * ...] for easier conversion to geoJSON object
 *
 * @param data Arrow.Table
 * @param by str, column name
 * @param params [{}, {}] result of Arrow.Table.toArray()
 * @returns [index:{props:propsValue}]
 */
function transform_data(data, by, params) {
  return data.reduce((a, v) => {
    a[v[[by]]] = params.reduce((res, x) => {
      res[x] = v[x];
      return res
    }, {});
    return a;
  }, {});
}

/**
 * convert an Arrow table to a geoJSON to be consumed by DeckGL GeoJSONLayer. Arrow table results
 * from cudf.DataFrame.toArrow() function.
 *
 * @param data Arrow.Table
 * @param by str, column name to be matched to the geoJSONProperty in the gjson object
 * @param properties [] of column names, properties to add from data to result geoJSON
 * @param geojsonProp str, property name in gjson object to be mapped to `by`
 * @returns geoJSON object consumable by DeckGL GeoJSONLayer
 */
function convertToGeoJSON(data, by, properties, geojsonProp) {
  data = transform_data(data.toArray(), by, properties);
  tempjson = [];
  gjson.features.forEach((val) => {
    if (val.properties[geojsonProp] in data) {
      tempjson.push({
        type: val.type,
        geometry: val.geometry,
        properties: { ...val.properties, ...data[val.properties[geojsonProp]] }
      })
    }
  });
  return tempjson;
}

/**
 * Computes cudf.Column eq operation on given column and value.
 *
 * @param column str, column name
 * @param value number
 * @returns Boolean mask result of column == value
 */
function filterByValue(column, value) { return df.get(column).eq(value); }

/**
 * Computes cudf.Column gt, lt and eq operations on given column and min, max values.
 *
 * @param column str, column name
 * @param min number
 * @param max number
 * @returns Boolean mask result of min <= column <= max
 */
function filterByRange(column, min, max) {
  boolmask_gt_eq = (df.get(column).gt(min)).logical_or(df.get(column).eq(min))
  boolmask_lt_eq = (df.get(column).lt(max)).logical_or(df.get(column).eq(max))
  return boolmask_gt_eq.logical_and(boolmask_lt_eq)
}

/**
 * Computes cudf.DataFrame.filter by parsing input query_dictionary, and performing filterByValue or
 * filterByRange as needed.
 *
 * @param query_dict {}, where key is column name, and value is either scalar value, or a range of
 *   scalar values
 * @param ignore str, column name which is to be ignored (if any) from query compute
 * @returns resulting cudf.DataFrame
 */
function parseQuery(query_dict, ignore) {
  if (ignore in query_dict) { delete query_dict[ignore]; }
  var t0 = performance.now();
  let boolmask = undefined;
  for (const [key, value] of Object.entries(query_dict)) {
    let temp_boolmask = undefined;
    if (df._accessor.names.includes(key)) {
      if (Array.isArray(value) && value.length == 2) {
        temp_boolmask = filterByRange(key, value[0], value[1]);
      } else if (typeof (value) == 'number') {
        temp_boolmask = filterByValue(key, value);
      }
    }
    if (temp_boolmask !== undefined) {
      boolmask = (boolmask == undefined) ? temp_boolmask : boolmask.logical_and(temp_boolmask);
    }
  }
  var res = '';
  if (boolmask == undefined) {
    res = df;
  } else {
    res = df.filter(boolmask);
  }
  var t1 = performance.now();
  console.log('Query ', query_dict, ', Time Taken:', (t1 - t0).toFixed(2), 'ms');
  return res;
}

/**
 * Computes cudf.DataFrame.groupby.
 *
 * @param df cudf.DataFrame object
 * @param by str, column name by which the groupby needs to be computed
 * @param agg str, aggregate function, one of the following:
 *   https://rapidsai.github.io/node-rapids/classes/cudf_src.groupbysingle.html#methods
 * @returns resulting cudf.DataFrame
 */
function groupby(df, by, agg) {
  var t0 = performance.now();
  const grp = df.groupBy({ by: by });
  const validAgg = (agg in grp && typeof grp[agg] == 'function');
  if (!validAgg) { return ({}); }
  var res = grp[agg]();
  var t1 = performance.now();
  console.log('Group by ', by, ', agg:', agg, 'Time Taken:', (t1 - t0).toFixed(2), 'ms');
  return res;
}

// socket connection
module.exports = (io) => {
  router.get('/', (req, res) => { res.send('ok'); })

  // SOCKET.IO paths
  io.on('connection', (socket) => {
    console.log('client connected');

    socket.on('disconnect', () => { console.log('client disconnected'); });

    socket.on('readMortgageData', (callback) => {
      if (datasetLoaded == 'mortgage') {
        console.log('dataset already loaded');
        socket.emit('data-points-update', df.toArrow().length);
        callback(true);
      } else {
        readMortgageData((cb) => {
          if (cb == true) {
            socket.emit('data-points-update', df.toArrow().length);
            callback(true);
          }
        });
      }
    });

    socket.on('readUberData', (callback) => {
      if (datasetLoaded == 'uber') {
        console.log('dataset already loaded');
        socket.emit('data-points-update', df.toArrow().length);
        callback(true);
      } else {
        readUberData((cb) => {
          if (cb == true) {
            socket.emit('data-points-update', df.toArrow().length);
            callback(true);
          }
        });
      }
    });

    socket.on('groupby', (by, agg, return_type, query_dict, callback) => {
      if (df == undefined) { callback({}); }
      console.log('\n\n');
      const res_df = parseQuery(query_dict, by);
      res = groupby(res_df, by, agg);
      var t0 = performance.now();
      res = res.sortValues({ [by]: { ascending: true, null_order: 'AFTER' } });
      console.log('Sort Dataframe by ', by, ', Time Taken:', (performance.now() - t0).toFixed(2), 'ms');
      if (return_type.type == 'geojson') {
        t0 = performance.now();
        const geojson = convertToGeoJSON(res.toArrow(), by, return_type.properties, return_type.geojsonProp);
        console.log(`convertToGeoJSON(${JSON.stringify({
          by,
          properties: return_type.properties,
          geojsonProp: return_type.geojsonProp
        })} Time Taken: ${(performance.now() - t0).toFixed(2)}ms`);
        callback(geojson);
      } else {
        callback(res.select([by, return_type.column]).toArrow().toArray());
      }
    });

    socket.on('get-dataset-size', (query_dict, callback) => {
      if (df == undefined) { callback({}); }
      socket.emit('data-points-update', parseQuery(query_dict, '').toArrow().length);
    });

    socket.on('delete-df', (callback) => {
      deleteData();
      callback(true);
    });
  })

  return router;
};
