// Copyright (c) 2020, NVIDIA CORPORATION.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import CUDF from './addon';
import { Column } from './column';
import { ColumnAccessor } from './column_accessor';

interface TableConstructor {
    readonly prototype: CUDFTable;
    new(props: {
        columns?: ReadonlyArray<Column> | null
    }): CUDFTable;
}

interface CUDFTable {

    [index: number]: any;

    readonly numColumns: number;
    readonly numRows: number;
    columns: ReadonlyArray<string> | null;
    _data: ColumnAccessor;
    
    getColumn(index: number): Column;
    _select(columns: ReadonlyArray<number> | ReadonlyArray<string> | null): CUDFTable;
    select_cols_by_index_array(column_indices: Array<number>): CUDFTable;
    select_cols_by_label_array(column_labels: Array<string>): CUDFTable
}

export type ColumnDictionary = {
    string: Column
};

export class Table extends (<TableConstructor> CUDF.Table) {
    constructor(columns_input: ColumnDictionary | ColumnAccessor | {}){
        switch (arguments.length){
            case 1: {
                let column_accessor: ColumnAccessor;
                if(columns_input instanceof ColumnAccessor){
                    column_accessor = columns_input;
                }else{
                    column_accessor = new ColumnAccessor(new Map(Object.entries(columns_input)));
                }
                
                let res: ReadonlyArray<Column> = column_accessor.columns();
                super({columns: res});
                this._data = column_accessor;
                //column names array
                this.columns = this._data.columns_as_array();
                break;
            }
            default: super({});
        }
    }

    select(columns: Array<number> | Array<string>): CUDFTable{
        const column_indices: Array<number> =  (columns as any[]).map((value) => {
            return this.transform_input_label(value);
        });
        
        const column_accessor = this._data.select_by_index_list_like(column_indices);
        return new Table(column_accessor);
        
    }

    slice(start: number | string, end: number | string): CUDFTable{
        start = this.transform_input_label(start);
        end = this.transform_input_label(end);
        const column_accessor = this._data.select_by_index_slice(start as number, end as number);
        return new Table(column_accessor);
    }

    transform_input_label(label: number | string): number{
        if(typeof(label) == "string" && this.columns?.includes(label)){
            label = this._data.label_to_index(label)
        }
        return label as number;
    }
    
}

Object.setPrototypeOf(CUDF.Table.prototype, new Proxy({}, {
    get(target: {}, p: any, table: any) {
        let i: string = p;
        switch (typeof p) {
            // @ts-ignore
            case 'string':
                if (table.columns.includes(i)) {
                    let column_index: number = table.columns.indexOf(i);
                    return table.getColumn(column_index);
                }
                break;
        }
        return Reflect.get(target, p, table);
    }
}));
