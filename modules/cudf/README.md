### Building
Run this command to build the module from the mono-repo root
```bash
npx lerna run build --scope="@nvidia/cudf" --stream

# To rebuild

npx lerna run rebuild --scope="@nvidia/cudf" --stream


# To run unit tests

npx lerna run test --scope="@nvidia/cudf"
```
