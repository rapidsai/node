# <div align="left"><img src="https://rapids.ai/assets/images/rapids_logo.png" width="90px"/>&nbsp; node-rapids cuSpatial - GPU-Accelerated Spatial and Trajectory Data Management and Analytics Library

JS bindings for [cuSpatial](https://github.com/rapidsai/cuspatial).

For detailed node-cuSpatial API, follow our [API Documentation](https://rapidsai.github.io/node/modules/cuspatial_src.html).

### Installation

`npm install @rapidsai/cuspatial`

Run this command to build the module from the mono-repo root

```bash
# To build
npx lerna run build --scope="@rapidsai/cuspatial" --stream

# To rebuild
npx lerna run rebuild --scope="@rapidsai/cuspatial" --stream

# To run unit tests
npx lerna run test --scope="@rapidsai/cuspatial"
```
