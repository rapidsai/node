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
    columns: ReadonlyArray<String> | null;

    getColumn(index: number): Column;

}

type ColumnDictionary = {
    string: Column
};

function noDuplicates(array: ReadonlyArray<string>){
    return new Set(array).size == array.length
}

export class Table extends (<TableConstructor> CUDF.Table) {
    constructor(column_dictionary: ColumnDictionary | {}){
        switch (arguments.length){
            case 1: {
                let res: ReadonlyArray<Column> = Object.values(column_dictionary);
                if(noDuplicates(Object.keys(column_dictionary))){
                    super({columns: res});
                    this.columns = Object.keys(column_dictionary);
                }else{
                    throw new Error("Column names should be unique");
                }                
                break;
            }
            default: super({});
        }
        
        
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
                return undefined;
        }
        return Reflect.get(target, p, table);
    }
}));
