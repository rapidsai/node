import runMiddleware from '../../components/server/run-middleware';
import { groupBy, numRows } from '../../components/server/cudf-utils';
const cache = require('../../components/server/cache')();

export default async function handler(req, res) {
  const [dataset, fn, by, aggregation] = req.query.param;
  await runMiddleware(dataset, req, res, cache);
  const query_dict = req.query.query_dict ? JSON.parse(req.query.query_dict) : undefined;

  // `query.columns` could be a string, or an Array of strings.
  // This flattens either case into a single Array, or defaults to null.
  const columns = req.query.columns ? [].concat(req.query.columns.split(',')) : null;

  // const columns = req.query.columns ? JSON.parse(req.query.columns) : undefined;
  console.log(columns);
  if (fn == "groupby") {
    groupBy(req[dataset], by, aggregation, columns, query_dict, res);
  } else if (fn == "numRows") {
    numRows(req[dataset], query_dict, res);
  }
}

export const config = {
  api: {
    externalResolver: true,
    bodyParse: false,
  },
}
