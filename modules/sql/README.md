# <div align="left"><img src="https://rapids.ai/assets/images/rapids_logo.png" width="90px"/>&nbsp; node-rapids GPU accelerated SQL engine

These js bindings allow for GPU accelerated SQL queries.

For example, the following snippet creates a DataFrame, then uses our SQL engine to select and query a DataFrame using the `SQLContext` module.

```javascript
var { Series, DataFrame, Int32 } = require("@rapidsai/cudf");
var { SQLContext } = require("@rapidsai/sql");

var a  = Series.new({type: new Int32(), data: [1, 2, 3]});
var b  = Series.new({type: new Int32(), data: [4, 5, 6]});
var df = new DataFrame({'a': a, 'b': b});

var sqlContext = new SQLContext();
sqlContext.createTable('test_table', df);

sqlContext.sql('SELECT a FROM test_table').result(); // [1, 2, 3]
```

We have also provided the `SQLCluster` module which allows one to run SQL queries on multiple GPUs.

```javascript
var { Series, DataFrame } = require("@rapidsai/cudf");
var { SQLCluster } = require("@rapidsai/sql");

var a  = Series.new(['foo', 'bar']);
var df = new DataFrame({'a': a});

var sqlCluster = await SQLCluster.init({numWorkers: 2});
await sqlCluster.createTable('test_table', df);

await sqlCluster.sql('SELECT a FROM test_table WHERE a LIKE \'%foo%\'');  // ['foo']
```

For detailed SQL API, [follow our API Documentation](https://rapidsai.github.io/node/modules/sql_src.html).

### Installation

`npm install @rapidsai/sql`

Run this command to build the module from the mono-repo root

```bash
# To build
npx lerna run build --scope="@rapidsai/sql" --stream

# To rebuild
npx lerna run rebuild --scope="@rapidsai/sql" --stream

# To run unit tests
npx lerna run test --scope="@rapidsai/sql"
```
