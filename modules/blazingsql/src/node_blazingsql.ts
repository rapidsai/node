import {DataFrame, Table} from '@rapidsai/cudf';

export declare function getTableScanInfo(logicalPlan: string): [string[], string[]];

export declare function runGenerateGraph(masterIndex: number,
                                         workerIds: any[],
                                         dataframes: DataFrame[],
                                         tableNames: string[],
                                         tableScans: string[],
                                         ctxToken: number,
                                         query: string,
                                         configOptions: Record<string, unknown>,
                                         sql: string,
                                         currentTimestamp: string): ExecutionGraph;

export declare function startExecuteGraph(executionGraph: ExecutionGraph, ctxToken: number): void;

export declare function getExecuteGraphResult(executionGraph: ExecutionGraph,
                                              ctxToken: number): {names: string[], tables: Table[]};

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
}

export declare class ExecutionGraph {
  constructor();
}
