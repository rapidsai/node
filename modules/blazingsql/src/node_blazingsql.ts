import {DataFrame, Table} from '@rapidsai/cudf';

export type ContextProps = {
  ralId: number; workerId: string; network_iface_name: string; ralCommunicationPort: number;
  workersUcpInfo: [];
  singleNode: boolean;
  configOptions: Record<string, unknown>;
  allocationMode: string;
  initialPoolSize: number | null;
  maximumPoolSize: number | null;
  enableLogging: boolean;
};

export declare class Context {
  constructor(props: ContextProps);
  readonly port: number;

  sql(masterIndex: number,
      workerIds: string[],
      dataframes: DataFrame[],
      tableNames: string[],
      tableScans: string[],
      ctxToken: number,
      query: string,
      configOptions: Record<string, unknown>,
      sql: string,
      currentTimestamp: string): {names: string[], tables: Table[]};

  getTableScanInfo(logicalPlan: string): [string[], string[]];
}
