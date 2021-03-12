# <div align="left"><img src="https://rapids.ai/assets/images/rapids_logo.png" width="90px"/>&nbsp; node-rapids cuDF - GPU DataFrames

The js bindings for [cuDF](https://github.com/rapidsai/cudf) provides an API that will be familiar to data engineers & data scientists, so they can use it to easily accelerate their workflows in a Javascript runtime environment, without going into the details of CUDA programming.

For example, the following snippet creates a series, then uses the GPU to run some calculations:

```javascript
var { Series, Int32 } = require("@nvidia/cudf");

var series1 = Series.new({ type: new Int32(), data: [1, 2, 3] });
console.log(series1.mean()); // 2
console.log(series1.max()); // 3
```

The following snippet creates a DataFrame, then uses the GPU to to run some calculations:

```javascript
var {
  DataFrame,
  DataType,
  Float64,
  GroupBy,
  Int32,
  Series,
} = require("@nvidia/cudf");

var a = Series.new({ type: new Int32(), data: [5, 4, 3, 2, 1, 0] });
var b = Series.new({ type: new Int32(), data: [0, 0, 1, 1, 2, 2] });
var df = new DataFrame({ a: a, b: b });
var grp = new GroupBy({ obj: df, by: ["a"] });

var groups = grp.getGroups();

console.log(...groups["keys"].get("a")); // [0,1,2,3,4,5]
console.log(...groups.values?.get("b")); // [2,2,1,1,0,0]
console.log(...groups["offsets"]); // [0,1,2,3,4,5,6]
```

For detailed node-cuDF API, follow our [API Documentation](https://rapidsai.github.io/node-rapids/modules/cudf_src.html).

### Installation

`npm install @nvidia/cudf`

Run this command to build the module from the mono-repo root

```bash
# To build
npx lerna run build --scope="@nvidia/cudf" --stream

# To rebuild
npx lerna run rebuild --scope="@nvidia/cudf" --stream

# To run unit tests
npx lerna run test --scope="@nvidia/cudf"
```
