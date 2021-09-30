# node-rapids GPU accelerated SQL engine

The js bindings for [BlazingSQL](https://github.com/BlazingDB/blazingsql) provides an API that allows for GPU accelerated SQL queries on [cuDF](https://github.com/rapidsai/cudf) DataFrames.

For example, the following snippet creates a DataFrame, then uses BlazingSQL to select and query a DataFrame using the `SQLContext` module.

```javascript
var { Series, DataFrame, Int32 } = require("@rapidsai/cudf");
var { SQLContext } = require("@rapidsai/sql");

var a  = Series.new({type: new Int32(), data: [1, 2, 3]});
var b  = Series.new({type: new Int32(), data: [4, 5, 6]});
var df = new DataFrame({'a': a, 'b': b});

var bc = new SQLContext();
bc.createTable('test_table', df);

bc.sql('SELECT a FROM test_table').result(); // [1, 2, 3]
```

We have also provided the `SQLCluster` module which allows one to run BlazingSQL queries on multiple child processes.

```javascript
var { Series, DataFrame } = require("@rapidsai/cudf");
var { SQLCluster } = require("@rapidsai/sql");

var a  = Series.new(['foo', 'bar']);
var df = new DataFrame({'a': a});

var bc = await SQLCluster.init({numWorkers: 2});
await bc.createTable('test_table', df);

await bc.sql('SELECT a FROM test_table WHERE a LIKE \'%foo%\'');  // ['foo']
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
