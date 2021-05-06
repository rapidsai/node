// Copyright (c) 2021, NVIDIA CORPORATION.
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

import {ColumnAccessor} from '../column_accessor';
import {DataFrame, SeriesMap} from '../data_frame';
import {Series} from '../series';
import {Table} from '../table';
import {ColumnsMap, TypeMap} from '../types/mappings';

export type JoinKey<
    P extends string,
    T extends TypeMap,
    Suffix extends string,
    TOn extends string[],
> = [P] extends TOn ? `${P}` : P extends keyof T ? `${P}${Suffix}` : `${P}`;

interface JoinProps<
  // clang-format off
  Lhs extends TypeMap,
  Rhs extends TypeMap,
  TOn extends(string & keyof Lhs & keyof Rhs)[],
  LSuffix extends string,
  RSuffix extends string
  // clang-format on
  > {
  lhs: DataFrame<Lhs>;
  rhs: DataFrame<Rhs>;
  on: TOn;
  lsuffix?: LSuffix;
  rsuffix?: RSuffix;
  nullEquality?: boolean;
}

// clang-format off
export class Join<
  Lhs extends TypeMap,
  Rhs extends TypeMap,
  TOn extends (string & keyof Lhs & keyof Rhs)[],
  LSuffix extends string,
  RSuffix extends string
> {
  // clang-format on
  private lhs: DataFrame<Lhs>;
  private rhs: DataFrame<Rhs>;
  private on: TOn;
  private lsuffix: LSuffix|'';
  private rsuffix: RSuffix|'';
  private nullEquality: boolean;

  constructor(props: JoinProps<Lhs, Rhs, TOn, LSuffix, RSuffix>) {
    const {lsuffix = '', rsuffix = '', nullEquality = true} = props;
    this.lhs                                                = props.lhs;
    this.rhs                                                = props.rhs;
    this.on                                                 = props.on;
    this.lsuffix                                            = lsuffix;
    this.rsuffix                                            = rsuffix;
    this.nullEquality                                       = nullEquality;
  }

  public left() {
    const {on} = this;

    // clang-format off
    const [lhsMap, rhsMap] = Table.leftJoin(
      this.lhs.select(on).asTable(),
      this.rhs.select(on).asTable(),
      this.nullEquality
    ).map((col) => Series.new(col));
    // clang-format on

    const lhs    = this.lhs.gather(lhsMap, true);
    const rhs    = this.rhs.drop(on).gather(rhsMap, true);
    const result = mergeResults(lhs, rhs, on, this.lsuffix, this.rsuffix);
    return new DataFrame(new ColumnAccessor(result));
  }

  public right() {
    return new Join({
             on: this.on,
             lhs: this.rhs,
             rhs: this.lhs,
             lsuffix: this.rsuffix,
             rsuffix: this.lsuffix
           })
      .left();
  }

  public inner() {
    const {on} = this;

    // clang-format off
    const [lhsMap, rhsMap] = Table.innerJoin(
      this.lhs.select(on).asTable(),
      this.rhs.select(on).asTable(),
      this.nullEquality
    ).map((col) => Series.new(col));
    // clang-format on

    const lhs    = this.lhs.gather(lhsMap, true);
    const rhs    = this.rhs.drop(on).gather(rhsMap, true);
    const result = mergeResults(lhs, rhs, on, this.lsuffix, this.rsuffix);
    return new DataFrame(new ColumnAccessor(result));
  }

  public outer() {
    const {on} = this;

    // clang-format off
    const [lhsMap, rhsMap] = Table.fullJoin(
      this.lhs.select(on).asTable(),
      this.rhs.select(on).asTable(),
      this.nullEquality
    ).map((col) => Series.new(col));
    // clang-format on

    const lhs = this.lhs.gather(lhsMap, true);
    const rhs = this.rhs.gather(rhsMap, true);

    // clang-format off
    // replace lhs nulls with rhs valids for each common column name
    const lhsValids = lhs.assign(on.reduce((cols, name) => ({
      ...cols, [name]: lhs.get(name).replaceNulls(rhs.get(name) as any)
    }), <any>{}) as SeriesMap<Lhs>);
    // clang-format on

    const result = mergeResults(lhsValids, rhs.drop(on), on, this.lsuffix, this.rsuffix);
    return new DataFrame(new ColumnAccessor(result));
  }

  public leftSemi() {
    const {on} = this;
    // clang-format off
    const lhsMap = Series.new(Table.leftSemiJoin(
      this.lhs.select(on).asTable(),
      this.rhs.select(on).asTable(),
      this.nullEquality
    ));
    // clang-format on
    return this.lhs.gather(lhsMap, true);
  }

  public leftAnti() {
    const {on} = this;
    // clang-format off
    const lhsMap = Series.new(Table.leftAntiJoin(
      this.lhs.select(on).asTable(),
      this.rhs.select(on).asTable(),
      this.nullEquality
    ));
    // clang-format on
    return this.lhs.gather(lhsMap, true);
  }
}

// clang-format off
function mergeResults<
  Lhs extends TypeMap,
  Rhs extends TypeMap,
  TOn extends string[],
  LSuffix extends string,
  RSuffix extends string
>(lhs: DataFrame<Lhs>, rhs: DataFrame<Rhs>, on: TOn, lsuffix: LSuffix, rsuffix: RSuffix) {

  type TResult = {
    [P in (string & keyof Lhs) as JoinKey<P, Rhs, LSuffix, TOn>]: Lhs[P]
  } & {
    [P in (string & keyof Rhs) as JoinKey<P, Lhs, RSuffix, TOn>]: Rhs[P]
  };

  // clang-format on
  function getColumns<T extends TypeMap>(lhs: DataFrame<T>, rhsNames: string[], suffix: string) {
    return lhs.names.reduce((cols, name) => {
      const newName = on.includes(name)         ? name
                      : rhsNames.includes(name) ? `${name}${suffix}`
                                                : name;
      cols[newName] = lhs.get(name)._col;
      return cols;
    }, <any>{}) as ColumnsMap<{[P in keyof TResult]: TResult[P]}>;
  }

  const lhsCols = getColumns(lhs, rhs.names, lsuffix);
  const rhsCols = getColumns(rhs, lhs.names, rsuffix);

  return (
    lsuffix == '' && rsuffix == ''  // <<< If no suffixes and overlapping names,
      ? {...rhsCols, ...lhsCols}    // <<< prefer the lhs cols over the rhs cols,
      : {...lhsCols, ...rhsCols}    // <<< otherwise prefer the right cols.
  );
}
