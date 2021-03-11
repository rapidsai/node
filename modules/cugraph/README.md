# <div align="left"><img src="https://rapids.ai/assets/images/rapids_logo.png" width="90px"/>&nbsp; rapids-js cuGraph - GPU Graph Analytics

The js bindings for [cuGraph](https://github.com/rapidsai/cugraph) is a collection of GPU accelerated graph algorithms that process data found in GPU DataFrames.

For detailed cuGraph-js API, follow our [API Documentation](https://rapidsai.github.io/rapids-js/modules/cugraph_src.html).

### Installation
`npm install @nvidia/cugraph`


Run this command to build the module from the mono-repo root
```bash
# To build
npx lerna run build --scope="@nvidia/cugraph" --stream

# To rebuild
npx lerna run rebuild --scope="@nvidia/cugraph" --stream

# To run unit tests
npx lerna run test --scope="@nvidia/cugraph"
```
