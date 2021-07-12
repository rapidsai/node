import {DataFrame, Table} from '@rapidsai/cudf';

import {BLAZINGSQL} from './addon';

// TODO: Move this somewhere else...
export const default_config: Record<string, unknown> = {
  JOIN_PARTITION_SIZE_THRESHOLD: 400000000,
  CONCATENATING_CACHE_NUM_BYTES_TIMEOUT: 100,
  MAX_JOIN_SCATTER_MEM_OVERHEAD: 500000000,
  MAX_NUM_ORDER_BY_PARTITIONS_PER_NODE: 8,
  NUM_BYTES_PER_ORDER_BY_PARTITION: 400000000,
  MAX_DATA_LOAD_CONCAT_CACHE_BYTE_SIZE: 400000000,
  FLOW_CONTROL_BYTES_THRESHOLD:
    18446744073709551615,  // https://en.cppreference.com/w/cpp/types/numeric_limits/max
  MAX_ORDER_BY_SAMPLES_PER_NODE: 10000,
  BLAZING_PROCESSING_DEVICE_MEM_CONSUMPTION_THRESHOLD: 0.9,
  BLAZING_DEVICE_MEM_CONSUMPTION_THRESHOLD: 0.6,
  BLAZ_HOST_MEM_CONSUMPTION_THRESHOLD: 0.75,
  BLAZING_LOGGING_DIRECTORY: 'blazing_log',
  BLAZING_CACHE_DIRECTORY: '/tmp/',
  BLAZING_LOCAL_LOGGING_DIRECTORY: 'blazing_log',
  MEMORY_MONITOR_PERIOD: 50,
  MAX_KERNEL_RUN_THREADS: 16,
  EXECUTOR_THREADS: 10,
  MAX_SEND_MESSAGE_THREADS: 20,
  LOGGING_LEVEL: 'trace',
  LOGGING_FLUSH_LEVEL: 'warn',
  ENABLE_GENERAL_ENGINE_LOGS: true,
  ENABLE_COMMS_LOGS: false,
  ENABLE_TASK_LOGS: false,
  ENABLE_OTHER_ENGINE_LOGS: false,
  LOGGING_MAX_SIZE_PER_FILE: 1073741824,  // 1 GB
  TRANSPORT_BUFFER_BYTE_SIZE: 1048576,    // 1 MB in bytes
  TRANSPORT_POOL_NUM_BUFFERS: 1000,
  PROTOCOL: 'AUTO',
  REQUIRE_ACKNOWLEDGE: false,
};

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
      workerIds: any[],
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
