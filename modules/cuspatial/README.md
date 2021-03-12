# <div align="left"><img src="https://rapids.ai/assets/images/rapids_logo.png" width="90px"/>&nbsp; node-rapids cuSpatial - GPU-Accelerated Spatial and Trajectory Data Management and Analytics Library

JS bindings for [cuSpatial](https://github.com/rapidsai/cuspatial).

For detailed cuSpatial-js API, follow our [API Documentation](https://rapidsai.github.io/node-rapids/modules/cuspatial_src.html).

### Installation

`npm install @nvidia/cuspatial`

Run this command to build the module from the mono-repo root

```bash
# To build
npx lerna run build --scope="@nvidia/cuspatial" --stream

# To rebuild
npx lerna run rebuild --scope="@nvidia/cuspatial" --stream

# To run unit tests
npx lerna run test --scope="@nvidia/cuspatial"
```
