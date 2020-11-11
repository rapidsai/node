import { Column } from "./column";

interface ColumnAccessorInterface {
    _data: Map<string, Column>

    insert(name: string, value: Column): void;
    select_by_label(key:string): ColumnAccessor | undefined;
    select_by_index(index: number): ColumnAccessor | undefined;
    select_by_index_slice(start: number, end: number): ColumnAccessor | undefined;
    select_by_index_list_like(index: Array<number>): ColumnAccessor | undefined;
    
    set_by_index(index: number, value: Column): void;
}

export class ColumnAccessor implements ColumnAccessorInterface{
    _data = new Map();

    constructor(data: Map<string, Column>){
        this._data = data
    }

    columns_as_array(): ReadonlyArray<string>{
        return Array.from(this._data.keys());
    }

    columns(): ReadonlyArray<Column>{
        return Array.from(this._data.values());
    }

    length(){
        return this._data.size;
    }

    insert(name: string, value: Column) {
        this._data.set(name, value);
    }

    select_by_label(key:string) {
        if(this._data.has(key)){
            let temp_val = this._data.get(key);
            if (temp_val != undefined){
                return new ColumnAccessor(new Map([[key, temp_val]]));
            }
        }
        return undefined;
    };

    select_by_index(index: number){
        const label: string = this.index_to_label(index);
        if(this._data.has(label)){
            return this.select_by_label(label)
        }
        return undefined;
    };

    select_by_index_slice(start: number, end: number){
        if(start >=0){
            return new ColumnAccessor(
                new Map(
                    Array.from(this._data).slice(start, end + 1)
                )
            )
        }
        return new ColumnAccessor(new Map());
    };

    select_by_index_list_like(index: Array<number>){
        let return_map = new Map(Array.from(this._data).filter(
            (_, i) => {
                return index.includes(i);
            }
        ))
        return new ColumnAccessor(return_map);
    };

    set_by_index(index: number, value: Column){
        this.insert(
            this.index_to_label(index),
            value
        )
    };

    label_to_index(label: string): number{
        return Array.from(this._data.keys()).indexOf(label);
    }

    index_to_label(index: number): string{
        return Array.from(this._data.keys())[index];
    }

    label_array_to_index_array(label: Array<string>): Array<number>{
        let return_array: Array<number> = new Array();
        for(let _label of label){
            if(this._data.has(_label)){
                return_array.push(this.label_to_index(_label));
            }
        }

        return return_array;
    }
}
