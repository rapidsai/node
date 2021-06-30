import {BLAZINGSQL} from './addon';

// TODO: Move this somewhere else...
export const default_config = {
  JOIN_PARTITION_SIZE_THRESHOLD: 500,
  CONCATENATING_CACHE_NUM_BYTES_TIMEOUT: 500,
  MAX_JOIN_SCATTER_MEM_OVERHEAD: 500,
  MAX_NUM_ORDER_BY_PARTITIONS_PER_NODE: 500,
  NUM_BYTES_PER_ORDER_BY_PARTITION: 500,
  MAX_DATA_LOAD_CONCAT_CACHE_BYTE_SIZE: 500,
  FLOW_CONTROL_BYTES_THRESHOLD: 500,
  MAX_ORDER_BY_SAMPLES_PER_NODE: 500,
  BLAZING_PROCESSING_DEVICE_MEM_CONSUMPTION_THRESHOLD: 500,
  BLAZING_DEVICE_MEM_CONSUMPTION_THRESHOLD: 500,
  BLAZ_HOST_MEM_CONSUMPTION_THRESHOLD: 500,
  BLAZING_LOGGING_DIRECTORY: 'test',
  MEMORY_MONITOR_PERIOD: 500,
  MAX_KERNEL_RUN_THREADS: 500,
  EXECUTOR_THREADS: 500,
  MAX_SEND_MESSAGE_THREADS: 500,
  LOGGING_LEVEL: 'string',
  LOGGING_FLUSH_LEVEL: 'string',
  ENABLE_GENERAL_ENGINE_LOGS: false,
  ENABLE_COMMS_LOGS: false,
  ENABLE_TASK_LOGS: false,
  ENABLE_OTHER_ENGINE_LOGS: false,
  LOGGING_MAX_SIZE_PER_FILE: 500,
  TRANSPORT_BUFFER_BYTE_SIZE: 500,
  TRANSPORT_POOL_NUM_BUFFERS: 500,
  PROTOCOL: 'false',
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
  sql(): void;
}

// eslint-disable-next-line @typescript-eslint/no-redeclare
export const Context: ContextConstructor = BLAZINGSQL.Context;
