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

    getColumnByIndex(index: number): Column;
    select(columns: ReadonlyArray<number> | ReadonlyArray<string> | null): CUDFTable;
    slice(start: number | string, end: number | string): CUDFTable;
    updateColumns(props: {
        columns?: ReadonlyArray<Column> | null
    }): void;

    _orderBy(ascending: Array<boolean>, na_position_flag: number): Column
}

export class Table extends (<TableConstructor>CUDF.Table) {
    constructor(props: {
        data?: ColumnAccessor,
    }) {
        if (!(props.data instanceof ColumnAccessor)) {
            props.data = new ColumnAccessor(new Map(Object.entries(typeof props.data === 'object' ? props.data || {} : {})));
        }
        super({ columns: props.data.columns });
        this._data = props.data;
    }

    get columns(): ReadonlyArray<string> {
        return this._data.names;
    }

    // TODO (bev) enum for na_position
    argsort(ascending: boolean | Array<boolean> = true, na_position: string = "last"): Column {
        return this._getSortedInds(ascending, na_position)
    }

    select(columns: Array<number> | Array<string>): CUDFTable {
        const column_indices: Array<number | undefined> = (columns as any[]).map((value) => {
            return this.transformInputLabel(value);
        });

        const column_accessor = this._data.selectByColumnIndices(column_indices);
        return new Table({ data: column_accessor });

    }

    slice(start: number | string, end: number | string): CUDFTable {
        return new Table({
            data: this._data.sliceByColumnIndices(
                this.transformInputLabel(start),
                this.transformInputLabel(end)
            )
        });
    }

    addColumn(name: string, column: Column) {
        this._data.insertByColumnName(name, column);
        super.updateColumns({ columns: this._data.columns });
    }

    getColumnByIndex(index: number): Column {
        if (typeof this.transformInputLabel(index) !== "undefined" && typeof index === "number") {
            return super.getColumnByIndex(index);
        }
        throw new Error("Column does not exist in the table: " + index);
    }

    getColumnByName(label: string): Column {
        let index = typeof label === "string" ? this.transformInputLabel(label) : undefined;
        if (typeof index !== "undefined") {
            return this.getColumnByIndex(index);
        }
        throw new Error("Column does not exist in the table: " + label);
    }

    drop(props: { columns: Array<string> }) {
        props.columns.forEach((value, _) => {
            this._data.removeByColumnName(value);
        })
        super.updateColumns({ columns: this._data.columns });
    }

    private transformInputLabel(label: number | string): number | undefined {
        if (typeof label === "string" && this.columns?.includes(label)) {
            return this._data.columnNameToColumnIndex(label)
        }
        else if (typeof label === "number" && label < this.columns?.length) {
            return label;
        }
        return undefined;
    }
 
    // TODO (bev) enum for na_position
    private _getSortedInds(ascending: boolean | Array<boolean>, na_position: string): Column {
        let na_position_flag: 0 | 1;
        if (ascending == true) {
            if (na_position == "last")
                na_position_flag = 0;
            else
                na_position_flag = 1;
        }
        else if (ascending == false) {
            if (na_position == "last")
                na_position_flag = 1;
            else
                na_position_flag = 0;
        }
        else {
            // TODO (bev) warning here 
            na_position_flag = 0
        }

        if (!Array.isArray(ascending)) {
            ascending = Array<boolean>(this.numColumns).fill(ascending);
        }

        return this._orderBy(ascending, na_position_flag);
    }

}

Object.setPrototypeOf(CUDF.Table.prototype, new Proxy({}, {
    get(target: {}, p: any, table: any) {
        let i: string = p;
        switch (typeof p) {
            // @ts-ignore
            case 'string':
                if (table.columns.includes(i)) {
                    return table.getColumnByName(i);
                }
                break;
        }
        return Reflect.get(target, p, table);
    }
}));
