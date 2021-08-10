# <div align="left"><img src="https://rapids.ai/assets/images/rapids_logo.png" width="90px"/>&nbsp; node-rapids cuGraph - GPU Graph Analytics

The js bindings for [cuGraph](https://github.com/rapidsai/cugraph) is a collection of GPU accelerated graph algorithms that process data found in GPU DataFrames.

For detailed node-cuGraph API, follow our [API Documentation](https://rapidsai.github.io/node/modules/cugraph_src.html).

### Installation

`npm install @rapidsai/cugraph`

Run this command to build the module from the mono-repo root

```bash
# To build
npx lerna run build --scope="@rapidsai/cugraph" --stream

# To rebuild
npx lerna run rebuild --scope="@rapidsai/cugraph" --stream

# To run unit tests
npx lerna run test --scope="@rapidsai/cugraph"
```
