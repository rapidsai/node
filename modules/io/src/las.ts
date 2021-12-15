import {DataFrame, Series} from '@rapidsai/cudf';

import {readLasTable} from './addon';

export class IO {
  public static readLas(path: string) {
    const {names, table} = readLasTable(path);
    return new DataFrame(names.reduce(
      (cols, name, i) => ({...cols, [name]: Series.new(table.getColumnByIndex(i))}), {}));
  }
}
