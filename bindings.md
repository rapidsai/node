# Bindings

The tables below show the bindings that have been implemented in `node-rapids`.

## cuDF

### Series

| cuDF                   | Numeric Series | String Series | Struct Series | List Series |
| ---------------------- | :------------: | :-----------: | :-----------: | :---------: |
| `abs()`                |       ➖       |      ➖       |      ➖       |     ➖      |
| `acos()`               |       ➖       |      ➖       |      ➖       |     ➖      |
| `add()`                |       ➖       |      ➖       |      ➖       |     ➖      |
| `all()`                |       ➖       |      ➖       |      ➖       |     ➖      |
| `any()`                |       ➖       |      ➖       |      ➖       |     ➖      |
| `append()`             |       ➖       |      ➖       |      ➖       |     ➖      |
| `applymap()`           |       ➖       |      ➖       |      ➖       |     ➖      |
| `argsort()`            |       ➖       |      ➖       |      ➖       |     ➖      |
| `as_index()`           |       ➖       |      ➖       |      ➖       |     ➖      |
| `as_mask()`            |       ➖       |      ➖       |      ➖       |     ➖      |
| `asin()`               |       ➖       |      ➖       |      ➖       |     ➖      |
| `astype()`             |       ➖       |      ➖       |      ➖       |     ➖      |
| `atan()`               |       ➖       |      ➖       |      ➖       |     ➖      |
| `ceil()`               |       ➖       |      ➖       |      ➖       |     ➖      |
| `clip()`               |       ➖       |      ➖       |      ➖       |     ➖      |
| `copy()`               |       ➖       |      ➖       |      ➖       |     ➖      |
| `corr()`               |       ➖       |      ➖       |      ➖       |     ➖      |
| `cos()`                |       ➖       |      ➖       |      ➖       |     ➖      |
| `count()`              |       ➖       |      ➖       |      ➖       |     ➖      |
| `cov()`                |       ➖       |      ➖       |      ➖       |     ➖      |
| `cummax()`             |       ➖       |      ➖       |      ➖       |     ➖      |
| `cummin()`             |       ➖       |      ➖       |      ➖       |     ➖      |
| `cumprod()`            |       ➖       |      ➖       |      ➖       |     ➖      |
| `cumsum()`             |       ➖       |      ➖       |      ➖       |     ➖      |
| `describe()`           |       ➖       |      ➖       |      ➖       |     ➖      |
| `diff()`               |       ➖       |      ➖       |      ➖       |     ➖      |
| `digitize()`           |       ➖       |      ➖       |      ➖       |     ➖      |
| `drop_duplicates()`    |       ➖       |      ➖       |      ➖       |     ➖      |
| `dropna()`             |       ➖       |      ➖       |      ➖       |     ➖      |
| `eq()`                 |       ➖       |      ➖       |      ➖       |     ➖      |
| `equals()`             |       ➖       |      ➖       |      ➖       |     ➖      |
| `exp()`                |       ➖       |      ➖       |      ➖       |     ➖      |
| `factorize()`          |       ➖       |      ➖       |      ➖       |     ➖      |
| `fillna()`             |       ➖       |      ➖       |      ➖       |     ➖      |
| `floor()`              |       ➖       |      ➖       |      ➖       |     ➖      |
| `floordiv()`           |       ➖       |      ➖       |      ➖       |     ➖      |
| `from_arrow()`         |       ➖       |      ➖       |      ➖       |     ➖      |
| `from_categorical()`   |       ➖       |      ➖       |      ➖       |     ➖      |
| `from_masked_array()`  |       ➖       |      ➖       |      ➖       |     ➖      |
| `from_pandas()`        |       ➖       |      ➖       |      ➖       |     ➖      |
| `ge()`                 |       ➖       |      ➖       |      ➖       |     ➖      |
| `groupby()`            |       ➖       |      ➖       |      ➖       |     ➖      |
| `gt()`                 |       ➖       |      ➖       |      ➖       |     ➖      |
| `hash_encode()`        |       ➖       |      ➖       |      ➖       |     ➖      |
| `hash_values()`        |       ➖       |      ➖       |      ➖       |     ➖      |
| `head()`               |       ➖       |      ➖       |      ➖       |     ➖      |
| `interleave_columns()` |       ➖       |      ➖       |      ➖       |     ➖      |
| `isin()`               |       ➖       |      ➖       |      ➖       |     ➖      |
| `isna()`               |       ➖       |      ➖       |      ➖       |     ➖      |
| `isnull()`             |       ➖       |      ➖       |      ➖       |     ➖      |
| `keys()`               |       ➖       |      ➖       |      ➖       |     ➖      |
| `kurt()`               |       ➖       |      ➖       |      ➖       |     ➖      |
| `kurtosis()`           |       ➖       |      ➖       |      ➖       |     ➖      |
| `label_encoding()`     |       ➖       |      ➖       |      ➖       |     ➖      |
| `le()`                 |       ➖       |      ➖       |      ➖       |     ➖      |
| `log()`                |       ➖       |      ➖       |      ➖       |     ➖      |
| `lt()`                 |       ➖       |      ➖       |      ➖       |     ➖      |
| `map()`                |       ➖       |      ➖       |      ➖       |     ➖      |
| `mask()`               |       ➖       |      ➖       |      ➖       |     ➖      |
| `max()`                |       ➖       |      ➖       |      ➖       |     ➖      |
| `mean()`               |       ➖       |      ➖       |      ➖       |     ➖      |
| `median()`             |       ➖       |      ➖       |      ➖       |     ➖      |
| `memory_usage()`       |       ➖       |      ➖       |      ➖       |     ➖      |
| `min()`                |       ➖       |      ➖       |      ➖       |     ➖      |
| `mod()`                |       ➖       |      ➖       |      ➖       |     ➖      |
| `mode()`               |       ➖       |      ➖       |      ➖       |     ➖      |
| `mul()`                |       ➖       |      ➖       |      ➖       |     ➖      |
| `nans_to_nulls()`      |       ➖       |      ➖       |      ➖       |     ➖      |
| `ne()`                 |       ➖       |      ➖       |      ➖       |     ➖      |
| `nlargest()`           |       ➖       |      ➖       |      ➖       |     ➖      |
| `notna()`              |       ➖       |      ➖       |      ➖       |     ➖      |
| `notnull()`            |       ➖       |      ➖       |      ➖       |     ➖      |
| `nsmallest()`          |       ➖       |      ➖       |      ➖       |     ➖      |
| `nunique()`            |       ➖       |      ➖       |      ➖       |     ➖      |
| `one_hot_encoding()`   |       ➖       |      ➖       |      ➖       |     ➖      |
| `pipe()`               |       ➖       |      ➖       |      ➖       |     ➖      |
| `pow()`                |       ➖       |      ➖       |      ➖       |     ➖      |
| `prod()`               |       ➖       |      ➖       |      ➖       |     ➖      |
| `product()`            |       ➖       |      ➖       |      ➖       |     ➖      |
| `quantile()`           |       ➖       |      ➖       |      ➖       |     ➖      |
| `radd()`               |       ➖       |      ➖       |      ➖       |     ➖      |
| `rank()`               |       ➖       |      ➖       |      ➖       |     ➖      |
| `reindex()`            |       ➖       |      ➖       |      ➖       |     ➖      |
| `rename()`             |       ➖       |      ➖       |      ➖       |     ➖      |
| `repeat()`             |       ✔️       |      ✔️       |      ✔️       |     ✔️      |
| `replace()`            |       ✔️       |      ✔️       |      ✔️       |     ✔️      |
| `reset_index()`        |       ✔️       |      ✔️       |      ✔️       |     ✔️      |
| `reverse()`            |       ✔️       |      ✔️       |      ✔️       |     ✔️      |
| `rfloordiv()`          |       ✔️       |      ✔️       |      ✔️       |     ✔️      |
| `rmod()`               |       ✔️       |      ✔️       |      ✔️       |     ✔️      |
| `rmul()`               |       ✔️       |      ✔️       |      ✔️       |     ✔️      |
| `rolling()`            |       ✔️       |      ✔️       |      ✔️       |     ✔️      |
| `round()`              |       ✔️       |      ✔️       |      ✔️       |     ✔️      |
| `rpow()`               |       ✔️       |      ✔️       |      ✔️       |     ✔️      |
| `rsub()`               |       ✔️       |      ✔️       |      ✔️       |     ✔️      |
| `rtruediv()`           |       ✔️       |      ✔️       |      ✔️       |     ✔️      |
| `sample()`             |       ✔️       |      ✔️       |      ✔️       |     ✔️      |
| `scale()`              |       ✔️       |      ✔️       |      ✔️       |     ✔️      |
| `scatter_by_map()`     |       ✔️       |      ✔️       |      ✔️       |     ✔️      |
| `searchsorted()`       |       ✔️       |      ✔️       |      ✔️       |     ✔️      |
| `set_index()`          |       ✔️       |      ✔️       |      ✔️       |     ✔️      |
| `set_mask()`           |       ✔️       |      ✔️       |      ✔️       |     ✔️      |
| `shift()`              |       ✔️       |      ✔️       |      ✔️       |     ✔️      |
| `sin()`                |       ✔️       |      ✔️       |      ✔️       |     ✔️      |
| `skew()`               |       ✔️       |      ✔️       |      ✔️       |     ✔️      |
| `sort_index()`         |       ✔️       |      ✔️       |      ✔️       |     ✔️      |
| `sort_values()`        |       ✔️       |      ✔️       |      ✔️       |     ✔️      |
| `sqrt()`               |       ✔️       |      ✔️       |      ✔️       |     ✔️      |
| `std()`                |       ✔️       |      ✔️       |      ✔️       |     ✔️      |
| `sub()`                |       ✔️       |      ✔️       |      ✔️       |     ✔️      |
| `sum()`                |       ✔️       |      ✔️       |      ✔️       |     ✔️      |
| `tail()`               |       ✔️       |      ✔️       |      ✔️       |     ✔️      |
| `take()`               |       ✔️       |      ✔️       |      ✔️       |     ✔️      |
| `tan()`                |       ✔️       |      ✔️       |      ✔️       |     ✔️      |
| `tile()`               |       ✔️       |      ✔️       |      ✔️       |     ✔️      |
| `to_array()`           |       ✔️       |      ✔️       |      ✔️       |     ✔️      |
| `to_arrow()`           |       ✔️       |      ✔️       |      ✔️       |     ✔️      |
| `to_dlpack()`          |       ✔️       |      ✔️       |      ✔️       |     ✔️      |
| `to_frame()`           |       ✔️       |      ✔️       |      ✔️       |     ✔️      |
| `to_gpu_array()`       |       ✔️       |      ✔️       |      ✔️       |     ✔️      |
| `to_hdf()`             |       ✔️       |      ✔️       |      ✔️       |     ✔️      |
| `to_json()`            |       ✔️       |      ✔️       |      ✔️       |     ✔️      |
| `to_pandas()`          |       ✔️       |      ✔️       |      ✔️       |     ✔️      |
| `to_string()`          |       ✔️       |      ✔️       |      ✔️       |     ✔️      |
| `truediv()`            |       ✔️       |      ✔️       |      ✔️       |     ✔️      |
| `unique()`             |       ✔️       |      ✔️       |      ✔️       |     ✔️      |
| `value_counts()`       |       ✔️       |      ✔️       |      ✔️       |     ✔️      |
| `values_to_string()`   |       ✔️       |      ✔️       |      ✔️       |     ✔️      |
| `var()`                |       ✔️       |      ✔️       |      ✔️       |     ✔️      |
| `where()`              |       ✔️       |      ✔️       |      ✔️       |     ✔️      |

### DataFrame

| cuDF                   | node-rapids |
| ---------------------- | :---------: |
| `acos()`               |     ✔️      |
| `add()`                |     ✔️      |
| `agg()`                |     ✔️      |
| `all()`                |     ✔️      |
| `any()`                |     ✔️      |
| `append()`             |     ✔️      |
| `apply_chunks()`       |     ✔️      |
| `apply_rows()`         |     ✔️      |
| `argsort()`            |     ✔️      |
| `as_gpu_matrix()`      |     ✔️      |
| `as_matrix()`          |     ✔️      |
| `asin()`               |     ✔️      |
| `assign()`             |     ✔️      |
| `astype()`             |     ✔️      |
| `atan()`               |     ✔️      |
| `clip()`               |     ✔️      |
| `copy()`               |     ✔️      |
| `corr()`               |     ✔️      |
| `cos()`                |     ✔️      |
| `count()`              |     ✔️      |
| `cov()`                |     ✔️      |
| `cummax()`             |     ✔️      |
| `cummin()`             |     ✔️      |
| `cumprod()`            |     ✔️      |
| `cumsum()`             |     ✔️      |
| `describe()`           |     ✔️      |
| `div()`                |     ✔️      |
| `drop()`               |     ✔️      |
| `drop_duplicates()`    |     ✔️      |
| `dropna()`             |     ✔️      |
| `equals()`             |     ✔️      |
| `exp()`                |     ✔️      |
| `fillna()`             |     ✔️      |
| `floordiv()`           |     ✔️      |
| `from_arrow()`         |     ✔️      |
| `from_pandas()`        |     ✔️      |
| `from_records()`       |     ✔️      |
| `hash_columns()`       |     ✔️      |
| `head()`               |     ✔️      |
| `info()`               |     ✔️      |
| `insert()`             |     ✔️      |
| `interleave_columns()` |     ✔️      |
| `isin()`               |     ✔️      |
| `isna()`               |     ✔️      |
| `isnull()`             |     ✔️      |
| `iteritems()`          |     ✔️      |
| `join()`               |     ✔️      |
| `keys()`               |     ✔️      |
| `kurt()`               |     ✔️      |
| `kurtosis()`           |     ✔️      |
| `label_encoding()`     |     ✔️      |
| `log()`                |     ✔️      |
| `mask()`               |     ✔️      |
| `max()`                |     ✔️      |
| `mean()`               |     ✔️      |
| `melt()`               |     ✔️      |
| `memory_usage()`       |     ✔️      |
| `merge()`              |     ✔️      |
| `min()`                |     ✔️      |
| `mod()`                |     ✔️      |
| `mode()`               |     ✔️      |
| `mul()`                |     ✔️      |
| `nans_to_nulls()`      |     ✔️      |
| `nlargest()`           |     ✔️      |
| `notna()`              |     ✔️      |
| `notnull()`            |     ✔️      |
| `nsmallest()`          |     ✔️      |
| `one_hot_encoding()`   |     ✔️      |
| `partition_by_hash()`  |     ✔️      |
| `pipe()`               |     ✔️      |
| `pivot()`              |     ✔️      |
| `pop()`                |     ✔️      |
| `pow()`                |     ✔️      |
| `prod()`               |     ✔️      |
| `product()`            |     ✔️      |
| `quantile()`           |     ✔️      |
| `quantiles()`          |     ✔️      |
| `query()`              |     ✔️      |
| `radd()`               |     ✔️      |
| `rank()`               |     ✔️      |
| `rdiv()`               |     ✔️      |
| `reindex()`            |     ✔️      |
| `rename()`             |     ✔️      |
| `repeat()`             |     ✔️      |
| `replace()`            |     ✔️      |
| `reset_index()`        |     ✔️      |
| `rfloordiv()`          |     ✔️      |
| `rmod()`               |     ✔️      |
| `rmul()`               |     ✔️      |
| `round()`              |     ✔️      |
| `rpow()`               |     ✔️      |
| `rsub()`               |     ✔️      |
| `rtruediv()`           |     ✔️      |
| `sample()`             |     ✔️      |
| `scatter_by_map()`     |     ✔️      |
| `searchsorted()`       |     ✔️      |
| `select_dtypes()`      |     ✔️      |
| `set_index()`          |     ✔️      |
| `shift()`              |     ✔️      |
| `sin()`                |     ✔️      |
| `skew()`               |     ✔️      |
| `sort_index()`         |     ✔️      |
| `sort_values()`        |     ✔️      |
| `sqrt()`               |     ✔️      |
| `stack()`              |     ✔️      |
| `std()`                |     ✔️      |
| `sub()`                |     ✔️      |
| `sum()`                |     ✔️      |
| `tail()`               |     ✔️      |
| `take()`               |     ✔️      |
| `tan()`                |     ✔️      |
| `tile()`               |     ✔️      |
| `to_arrow()`           |     ✔️      |
| `to_csv()`             |     ✔️      |
| `to_dlpack()`          |     ✔️      |
| `to_feather()`         |     ✔️      |
| `to_hdf()`             |     ✔️      |
| `to_json()`            |     ✔️      |
| `to_orc()`             |     ✔️      |
| `to_pandas()`          |     ✔️      |
| `to_parquet()`         |     ✔️      |
| `to_records()`         |     ✔️      |
| `to_string()`          |     ✔️      |
| `transpose()`          |     ✔️      |
| `truediv()`            |     ✔️      |
| `unstack()`            |     ✔️      |
| `update()`             |     ✔️      |
| `var()`                |     ✔️      |
| `where()`              |     ✔️      |

### GroupBy

| cuDF              | node-rapids |
| ----------------- | :---------: |
| `agg()`           |     ➖      |
| `aggregate()`     |     ➖      |
| `apply()`         |     ➖      |
| `apply_grouped()` |     ➖      |
| `nth()`           |     ➖      |
| `pipe()`          |     ➖      |
| `rolling()`       |     ➖      |
| `size()`          |     ➖      |
