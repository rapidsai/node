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

const { Router } = require('express');
const { RecordBatchStreamWriter } = require('apache-arrow');
const { performance } = require('perf_hooks');

module.exports = () => {
  return Router().get('/groupby/:by/:aggregation', (req, res) => {

    const { by, aggregation } = req.params;

    // `query.columns` could be a string, or an Array of strings.
    // This flattens either case into a single Array, or defaults to null.
    const columns = req.query.columns ? [].concat(req.query.columns) : null;

    // Perf: only include the subset of columns we want to return in `df.groupBy()[agg]()`
    const colsToUse = columns || req.uberTrips.names.filter((n) => n !== by);

    let t0;

    try {

      console.log('\n\n');
      t0 = performance.now();

      let trips = req.uberTrips
        .select([by, ...colsToUse])
        .groupBy({ by })[aggregation]();

      console.log(`trips.groupBy({by:${by}}).${aggregation}() Time Taken: ${(performance.now() - t0).toFixed(2)}ms`);

      t0 = performance.now();

      trips = trips.sortValues({
        [by]: {
          ascending: true,
          null_order: 'AFTER'
        }
      });

      console.log(`trips.sortValues({${by}}) Time Taken: ${(performance.now() - t0).toFixed(2)}ms`);
      t0 = performance.now();

      const tracts = req.uberTracts.gather(trips.get(by))
        .assign((columns || []).reduce((cols, col) => ({
          ...cols, [col]: trips.get(col)
        }), {}));

      console.log(`tracts.gather(trips.get(${by})) Time Taken: ${(performance.now() - t0).toFixed(2)}ms`);

      const writer = new RecordBatchStreamWriter();
      writer.pipe(res.status(200).type('application/octet-stream'));
      writer.write(tracts.toArrow());
      writer.close();

    } catch (e) {
      res.status(500).send(e ? `${(e.stack || e.message)}` : 'Unknown error');
    }
  });
};
