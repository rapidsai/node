import {BLAZINGSQL} from './addon';

export type ContextProps = {
  ralId?: number; workderId: string; network_iface_name: string; ralCommunicationPort: number;
  workersUcpInfo?: ReadonlyArray<number>;  // TODO: Fix.
  singleNode?: boolean;
  allocationMode?: string;
  initialPoolSize?: number | null;
  maximumPoolSize?: number | null;
  enableLogging?: boolean;
};

interface ContextConstructor {
  readonly prototype: Context;
  new(props: ContextProps): Context;
}

export interface Context {
  readonly ralId: number;
}

// eslint-disable-next-line @typescript-eslint/no-redeclare
export const Context: ContextConstructor = BLAZINGSQL.Context;
