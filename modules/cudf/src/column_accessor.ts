import { Column } from "./column";

interface ColumnAccessorInterface {
    _data: ReadonlyMap<string, Column>;

    insertByColumnName(name: string, value: Column): void;
    removeByColumnName(name: string): void;

    selectByColumnName(key:string | undefined): ColumnAccessor | undefined;
    sliceByColumnLabels(start: string, end: string): ColumnAccessor | undefined;
    selectByColumnNames(key: Array<string>): ColumnAccessor | undefined;

    selectByColumnIndex(index: number): ColumnAccessor | undefined;
    sliceByColumnIndices(start: number, end: number): ColumnAccessor | undefined;
    selectByColumnIndices(index: Array<number>): ColumnAccessor | undefined;
    
    columnNameToColumnIndex(label: string): number | undefined;
    columnIndexToColumnName(index: number): string | undefined;
    columnNamesToColumnIndices(label: Array<string>): Array<number>;
}

export class ColumnAccessor implements ColumnAccessorInterface{
    _data = new Map();
    #_labels_array = new Array();
    #_labels_to_indices: Map<string, number> = new Map();

    set data(value: Map<String, Column>){
        this._data = value;
        this.#_labels_array = Array.from(this._data.keys());
        this.#_labels_array.forEach((val, index)=>
            this.#_labels_to_indices.set(val, index)
        );
    }

    private addData(name: string, value: Column){
        this._data.set(name, value);
        this.#_labels_array.push(name);
        this.#_labels_to_indices.set(name, this.#_labels_array.indexOf(name));
    }

    private removeData(name: string){
        if(this._data.has(name)){
            this._data.delete(name);
            this.#_labels_to_indices.delete(name);
            this.#_labels_array  = this.#_labels_array.filter(
                x => x !== name
            );
        }
    }

    constructor(data: Map<string, Column>){
        this.data = data;
    }

    get names(): ReadonlyArray<string>{
        return this.#_labels_array;
    }

    get columns(): ReadonlyArray<Column>{
        return Array.from(this._data.values());
    }

    get length(){
        return this._data.size;
    }

    insertByColumnName(name: string, value: Column) {
        this.addData(name, value);
    }
    
    removeByColumnName(name: string) {
        this.removeData(name);
    }

    selectByColumnName(key:string | undefined) {
        if(key != undefined && this._data.has(key)){
            let temp_val = this._data.get(key);
            if (temp_val != undefined){
                return new ColumnAccessor(new Map([[key, temp_val]]));
            }
        }
        return new ColumnAccessor(new Map());
    };

    sliceByColumnLabels(start: string, end: string){
        return this.sliceByColumnIndices(
            this.columnNameToColumnIndex(start), this.columnNameToColumnIndex(end)
        );
    };

    selectByColumnNames(key: Array<string>) {
        let return_map = new Map(Array.from(this._data).filter(
            (x, _) => {
                return key.includes(x[0]);
            }
        ))
        return new ColumnAccessor(return_map);
    };

    selectByColumnIndex(index: number){
        const label = this.columnIndexToColumnName(index);
        return this.selectByColumnName(label);
    };

    sliceByColumnIndices(start: number | undefined, end: number | undefined){
        let _start: number = (typeof start == "undefined")? 0: start as number;
        let _end = (typeof end == "undefined")? this.#_labels_array.length: end as number;
        
        if(_start >= 0){
            return new ColumnAccessor(
                new Map(
                    Array.from(this._data).slice(_start, _end + 1)
                )
            )
        }
        return new ColumnAccessor(new Map());
    };

    selectByColumnIndices(index: Array<number | undefined>){
        let return_map = new Map(Array.from(this._data).filter(
            (x, _) => {
                let temp_val = this.columnNameToColumnIndex(x[0]);
                if(temp_val != undefined){
                    return index.includes(temp_val);
                }
                return false;
            }
        ))
        return new ColumnAccessor(return_map);
    };

    columnNameToColumnIndex(label: string): number | undefined{
        return this.#_labels_to_indices.get(label);
    }

    columnIndexToColumnName(index: number): string | undefined{
        return this.#_labels_array[index];
    }

    columnNamesToColumnIndices(label: Array<string>): Array<number>{
        let return_array: Array<number> = new Array();
        for(let _label of label){
            let temp_index = this.columnNameToColumnIndex(_label);
            if(this._data.has(_label) && temp_index != undefined){
                return_array.push(temp_index);
            }
        }

        return return_array;
    }
}
