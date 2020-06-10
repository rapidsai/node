import cudf
import pyarrow as pa
df = cudf.read_parquet("./data/census_data_minimized.parquet/temp.parquet")
table = df.to_arrow()
writer = pa.RecordBatchStreamWriter('data/census_data_minimized.arrow', table.schema)
writer.write_table(pa.Table.from_batches(table.to_batches(max_chunksize=len(table) / 1000)))
writer.close()

