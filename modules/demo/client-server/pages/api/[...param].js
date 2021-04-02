import runMiddleware from '../../components/server/run-middleware';
import groupBy from '../../components/server/cudf-groupby';
const cache = require('../../components/server/cache')();

export default async function handler(req, res) {
  await runMiddleware(req, res, cache);
  const [dataset, fn, by, aggregation] = req.query.param;

  if (dataset == "uber" && fn == "groupby") {
    await groupBy(req.uberTrips, by, aggregation, req.query.columns, JSON.parse(req.query.query_dict), res);
  }
}

export const config = {
  api: {
    externalResolver: true,
    bodyParse: false,
  },
}
