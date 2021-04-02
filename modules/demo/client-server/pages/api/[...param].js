import runMiddleware from '../../components/server/run-middleware';
import { groupBy, numRows } from '../../components/server/cudf-utils';
const cache = require('../../components/server/cache')();

export default async function handler(req, res) {
  await runMiddleware(req, res, cache);
  const [dataset, fn, by, aggregation] = req.query.param;
  const query_dict = req.query.query_dict ? JSON.parse(req.query.query_dict) : undefined;

  if (dataset == "uber" && fn == "groupby") {
    await groupBy(req.uberTrips, by, aggregation, req.query.columns, query_dict, res);
  } else if (dataset == "uber" && fn == "numRows") {
    await numRows(req.uberTrips, query_dict, res);
  }
}

export const config = {
  api: {
    externalResolver: true,
    bodyParse: false,
  },
}
