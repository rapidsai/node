// Copyright (c) 2021, NVIDIA CORPORATION.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
