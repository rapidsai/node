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

import { CUDABuffer } from './cuda';

/** @ignore */
export const isNumber = (x: any): x is number => typeof x === 'number';
/** @ignore */
export const isBigInt = (x: any): x is bigint => typeof x === 'bigint';
/** @ignore */
export const isBoolean = (x: any): x is boolean => typeof x === 'boolean';
/** @ignore */
export const isFunction = (x: any): x is Function => typeof x === 'function';
/** @ignore */
export const isObject = (x: any): x is Object => x != null && Object(x) === x;

/** @ignore */
export const isPromise = <T = any>(x: any): x is PromiseLike<T> => {
    return isObject(x) && isFunction(x.then);
};

/** @ignore */
export const isIterable = <T = any>(x: any): x is Iterable<T> => {
    return isObject(x) && isFunction(x[Symbol.iterator]);
};

/** @ignore */
export const isAsyncIterable = <T = any>(x: any): x is AsyncIterable<T> => {
    return isObject(x) && isFunction(x[Symbol.asyncIterator]);
};

/** @ignore */
export const isArrayLike = <T = any>(x: any): x is ArrayLike<T> => {
    return isObject(x) && isNumber(x['length']);
};

/** @ignore */
export const isArrayBuffer = (x: any): x is ArrayBuffer => {
    return x && x.constructor && x.constructor.name === 'ArrayBuffer';
};

/** @ignore */
export const isArrayBufferView = ArrayBuffer.isView;

/** @ignore */
export const isCUDABuffer = (x: any): x is CUDABuffer => {
    return x && x.constructor && x.constructor.name === 'CUDABuffer';
};

/** @ignore */
export const isIteratorResult = <T = any>(x: any): x is IteratorResult<T> => {
    return isObject(x) && ('done' in x) && ('value' in x);
};

export function cachedLookup<TResult>(field: string, getValue: (_: any) => TResult) {
    const _prop = `_${field}`;
    return function(this: any): TResult {
        if (typeof this[_prop] === 'undefined') {
            this[_prop] = getValue(this.id);
        }
        return this[_prop];
    }
}

export function cachedEnumLookup<TResult>(field: string, attr: any, getValue: (_: any, attr: any) => TResult) {
    const _prop = `_${field}`;
    return function(this: any): TResult {
        if (typeof this[_prop] === 'undefined') {
            this[_prop] = getValue(this.id, attr);
        }
        return this[_prop];
    }
}
