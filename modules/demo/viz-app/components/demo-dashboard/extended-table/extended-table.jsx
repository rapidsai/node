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

import React from 'react';
import { useTable, useSortBy, usePagination } from 'react-table';
import styles from './extended-table.module.css';

export default function ExtendedTable({ cols, data }) {
  const columns = React.useMemo(
    () => cols, []);

  const {
    getTableProps,
    getTableBodyProps,
    headerGroups,
    page,
    prepareRow,

    canPreviousPage,
    canNextPage,
    pageOptions,
    pageCount,
    gotoPage,
    nextPage,
    previousPage,
    setPageSize,
    state: { pageIndex, pageSize },
  } = useTable({
    columns,
    data,
    initialState: { pageIndex: 0 },
  }, useSortBy, usePagination);

  return (
    <>
      <table className={styles.table} {...getTableProps()}>
        <thead>
          {headerGroups.map(headerGroup => (
            <tr {...headerGroup.getHeaderGroupProps()}>
              {headerGroup.headers.map(column => (
                <th className={styles.th} {...column.getHeaderProps(column.getSortByToggleProps())}>
                  {column.render('Header')}
                  <span>
                    {column.isSorted
                      ? column.isSortedDesc
                        ? ' ▼'
                        : ' ▲'
                      : ''}
                  </span>
                </th>
              ))}
            </tr>
          ))}
        </thead>
        <tbody {...getTableBodyProps()}>
          {page.map(
            (row, i) => {
              prepareRow(row);
              return (
                <tr className={i % 2 != 0 ? styles.grey : ''} {...row.getRowProps()}>
                  {row.cells.map(cell => {
                    return (
                      <td className={styles.td} {...cell.getCellProps()}>{cell.render('Cell')}</td>
                    )
                  })}
                </tr>
              )
            }
          )}
        </tbody>
      </table>
      <div className={styles.spacer}>
        <div />
        <div className={styles.pagination}>
          <select
            className={styles.select}
            value={pageSize}
            onChange={e => {
              setPageSize(Number(e.target.value))
            }}
          >
            {[10, 20, 30, 40, 50].map(pageSize => (
              <option key={pageSize} value={pageSize}>
                Show {pageSize}
              </option>
            ))}
          </select>
          <div style={{ paddingRight: 8 }}>▼</div>
          <div className={"textButton"} style={{ paddingRight: 5 }} onClick={() => gotoPage(0)} disabled={!canPreviousPage}>
            {'<<'}
          </div>{' '}
          <div className={"textButton"} style={{ paddingRight: 5 }} onClick={() => previousPage()} disabled={!canPreviousPage}>
            {'<'}
          </div>{' '}
          <span style={{ paddingRight: 5 }}>
            <strong>
              {pageIndex + 1} of {pageOptions.length}
            </strong>{' '}
          </span>
          <div className={"textButton"} style={{ paddingRight: 5 }} onClick={() => nextPage()} disabled={!canNextPage}>
            {'>'}
          </div>{' '}
          <div className={"textButton"} onClick={() => gotoPage(pageCount - 1)} disabled={!canNextPage}>
            {'>>'}
          </div>{' '}
        </div>
      </div>
    </>
  )
}
