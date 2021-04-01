import runMiddleware from '../../components/server/run-middleware';
import groupBy from '../../components/server/cudf-groupby';
const cache = require('../../components/server/cache')();
const { RecordBatchStreamWriter } = require('apache-arrow');

export default async function handler(req, res) {
  await runMiddleware(req, res, cache);
  const [dataset, fn, by, aggregation] = req.query.param;

  if (dataset == "uber") {
    await groupBy(req.uberTrips, by, aggregation, req.query.columns, res);
  }
}

export const config = {
  api: {
    externalResolver: true,
    bodyParse: false,
  },
}
