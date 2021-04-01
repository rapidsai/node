import { resolve } from 'path';

const { performance } = require('perf_hooks');
const { RecordBatchStreamWriter } = require('apache-arrow');

export default async function groupBy(df, by, aggregation, columns, res) {
  // `query.columns` could be a string, or an Array of strings.
  // This flattens either case into a single Array, or defaults to null.
  columns = columns ? [].concat(columns) : null;

  // Perf: only include the subset of columns we want to return in `df.groupBy()[agg]()`
  const colsToUse = columns || df.names.filter((n) => n !== by);

  let t0;
  try {

    console.log('\n\n');
    t0 = performance.now();

    let trips = df.groupBy({ by })[aggregation]();

    console.log(`trips.groupBy({by:${by}}).${aggregation}() Time Taken: ${(performance.now() - t0).toFixed(2)}ms`);

    t0 = performance.now();

    trips = trips.sortValues({
      [by]: {
        ascending: true,
        null_order: 'AFTER'
      }
    });

    console.log(`trips.sortValues({${by}}) Time Taken: ${(performance.now() - t0).toFixed(2)}ms`);

    res.writeHead(200, 'Ok', { 'Content-Type': 'application/octet-stream' });
    const writer = new RecordBatchStreamWriter();
    writer.pipe(res);
    writer.write(trips.toArrow());
    writer.close();
  } catch (e) {
    res.status(500).send(e ? `${(e.stack || e.message)}` : 'Unknown error');
  }
  // t0 = performance.now();

  // const tracts = req.uberTracts.gather(trips.get(by))
  //   .assign((columns || []).reduce((cols, col) => ({
  //     ...cols, [col]: trips.get(col)
  //   }), {}));

  // console.log(`tracts.gather(trips.get(${by})) Time Taken: ${(performance.now() - t0).toFixed(2)}ms`);
}
