const { performance } = require('perf_hooks');
const { RecordBatchStreamWriter } = require('apache-arrow');
//A module method to pipe between streams and generators forwarding errors and properly cleaning up and provide a callback when the pipeline is complete.
const pipeline = require('util').promisify(require('stream').pipeline);


/**
 * Computes cuDF.Column eq operation on given column and value.
 *
 * @param df cuDF.DataFrame
 * @param column str, column name
 * @param value number
 * @returns Boolean mask result of column == value
 */
function filterByValue(df, column, value) { return df.get(column).eq(value); }

/**
 * Computes cuDF.Column gt, lt and eq operations on given column and min, max values.
 *
 * @param df cuDF.DataFrame
 * @param column str, column name
 * @param minMaxArray Array
 * @returns Boolean mask result of min <= column <= max
 */
function filterByRange(df, column, minMaxArray) {
  const [min, max] = minMaxArray;
  const boolmask_gt_eq = (df.get(column).gt(min)).logical_or(df.get(column).eq(min))
  const boolmask_lt_eq = (df.get(column).lt(max)).logical_or(df.get(column).eq(max))
  return boolmask_gt_eq.logical_and(boolmask_lt_eq)
}

/**
 * Computes cuDF.DataFrame.filter by parsing input query_dictionary, and performing filterByValue or
 * filterByRange as needed.
 *
 * @param query_dict {}, where key is column name, and value is either scalar value, or a range of
 *   scalar values
 * @param ignore str, column name which is to be ignored (if any) from query compute
 * @returns resulting cuDF.DataFrame
 */
function parseQuery(df, query_dict, ignore = null) {
  let t0 = performance.now();
  if (ignore && ignore in query_dict) { delete query_dict[ignore]; }

  let boolmask = undefined;

  Object.keys(query_dict)
    .filter(key => df._accessor.names.includes(key))
    .forEach(key => {
      //check if query is range of values
      if (Array.isArray(query_dict[key]) && query_dict[key].length == 2) {
        boolmask = boolmask ? boolmask.logical_and(filterByRange(df, key, query_dict[key])) : filterByRange(df, key, query_dict[key]);
      }
      //check if query is a number
      else if (typeof (query_dict[key]) == "number") {
        boolmask = boolmask ? boolmask.logical_and(filterByValue(df, key, query_dict[key])) : filterByValue(df, key, query_dict[key]);
      }
    })


  if (boolmask) { df = df.filter(boolmask); }
  console.log(`Query ${JSON.stringify(query_dict)}  Time Taken: ${(performance.now() - t0).toFixed(2)}ms`);
  return df;
}

async function groupBy(df, by, aggregation, columns, query_dict, res) {

  console.log('\n\n');

  // filter the dataframe as the query_dict & ignore the by column (to make sure chart selection doesn't filter self)
  if (query_dict && Object.keys(query_dict).length > 0) {
    df = parseQuery(df, query_dict, by);
  }

  // `query.columns` could be a string, or an Array of strings.
  // This flattens either case into a single Array, or defaults to null.
  columns = columns ? [].concat(columns) : null;

  // Perf: only include the subset of columns we want to return in `df.groupBy()[agg]()`
  const colsToUse = columns || df.names.filter((n) => n !== by);

  let t0;
  try {

    t0 = performance.now();

    let trips = df.select([by, ...colsToUse]).groupBy({ by })[aggregation]();

    console.log(`trips.groupBy({by:${by}}).${aggregation}() Time Taken: ${(performance.now() - t0).toFixed(2)}ms`);

    t0 = performance.now();

    trips = trips.sortValues({
      [by]: {
        ascending: true,
        null_order: 'AFTER'
      }
    });

    console.log(`trips.sortValues({${by}}) Time Taken: ${(performance.now() - t0).toFixed(2)}ms`);

    //stream data to client, where `arrow.Table(fetch(url, {method:'GET'}))` consumes it
    await pipeline(
      RecordBatchStreamWriter.writeAll(trips.toArrow()).toNodeStream(),
      res.writeHead(200, 'Ok', { 'Content-Type': 'application/octet-stream' })
    );

  } catch (e) {
    res.status(500).send(e ? `${(e.stack || e.message)}` : 'Unknown error');
  }

  // TODO in future, when binary data input to geoJSONLayer in deck.gl is implemented,
  // offload the geojson geometry gather calculations to the server-side

  // t0 = performance.now();

  // const tracts = req.uberTracts.gather(trips.get(by))
  //   .assign((columns || []).reduce((cols, col) => ({
  //     ...cols, [col]: trips.get(col)
  //   }), {}));

  // console.log(`tracts.gather(trips.get(${by})) Time Taken: ${(performance.now() - t0).toFixed(2)}ms`);
}


async function numRows(df, query_dict, res) {

  console.log('\n\n NumRows');

  // filter the dataframe as the query_dict & ignore the by column (to make sure chart selection doesn't filter self)
  if (query_dict && Object.keys(query_dict).length > 0) {
    df = parseQuery(df, query_dict);
  }

  res.send(df.numRows);
}


export { groupBy, numRows }
