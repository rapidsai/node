export interface ReadParquetOptions {
  sources: string[];
  /** The list of columns to read */
  columns?: string[];
  // TODO row_groups;
  /** The number of rows to skip from the start of the file */
  skipRows?: number;
  /** The total number of rows to read */
  numRows?: number;
  /** Return string columns as GDF_CATEGORY dtype */
  stringsToCategorical?: boolean;
  /**
   * If true and dataset has custom PANDAS schema metadata, ensure that index columns are also
   * loaded.
   */
  usePandasMetadata?: boolean;
}
