# Build instructions (within node-rapids container)

```bash
yarn
yarn dev
```

Uber dashboard at: `http://localhost:3000/dashboard/uber`

Api route that is failing: `http://localhost:3000/api/uber/groupby/sourceid/mean/[travel_time,%20sourceid]`, route just contains a single file to include `@rapidsai/cudf`
