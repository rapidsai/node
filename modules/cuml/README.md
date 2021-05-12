# <div align="left"><img src="https://rapids.ai/assets/images/rapids_logo.png" width="90px"/>&nbsp; node-rapids cuML - GPU Machine Learning Algorithms

The js bindings for [cuML](https://github.com/rapidsai/cuml) is a suite of libraries that implement machine learning algorithms and mathematical primitives that share compatible APIs with other RAPIDS projects.

For detailed node-cuML API, follow our [API Documentation](https://rapidsai.github.io/node-rapids/modules/cuml_src.html).

### Installation

`npm install @rapidsai/cuml`

Run this command to build the module from the mono-repo root

```bash
# To build
npx lerna run build --scope="@rapidsai/cuml" --stream

# To rebuild
npx lerna run rebuild --scope="@rapidsai/cuml" --stream

# To run unit tests
npx lerna run test --scope="@rapidsai/cuml"
```
