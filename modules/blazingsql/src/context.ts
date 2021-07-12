import {DataFrame, Table} from '@rapidsai/cudf';
import {BLAZINGSQL} from './addon';

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

interface ContextConstructor {
  readonly prototype: Context;
  new(props: ContextProps): Context;
}

export interface Context {
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

// eslint-disable-next-line @typescript-eslint/no-redeclare
export const Context: ContextConstructor = BLAZINGSQL.Context;
