import React from 'react';
import { makeStyles } from '@material-ui/core/styles';
import Paper from '@material-ui/core/Paper';
import Table from '@material-ui/core/Table';
import TableBody from '@material-ui/core/TableBody';
import TableCell from '@material-ui/core/TableCell';
import TableContainer from '@material-ui/core/TableContainer';
import TableHead from '@material-ui/core/TableHead';
import TablePagination from '@material-ui/core/TablePagination';
import TableRow from '@material-ui/core/TableRow';
import Typography from '@material-ui/core/Typography';

const columns = [
  { id: 'id', label: 'ID', minWidth: 0, },
  { id: 'revid', label: 'Rev ID', minWidth: 0, },
  { id: 'url', label: 'URL', minWidth: 0, },
  { id: 'title', label: 'Title', minWidth: 0, },
  { id: 'text', label: 'Text', minWidth: 1000 }
];

function createData(id, revid, url, title, text) {
  return { id, revid, url, title, text };
}

function formatData(data) {
  if (Object.keys(data).length === 0) {
    return [];
  }

  let rows = [];
  data['title'].forEach((_, idx) => {
    const id = data['id'][idx];
    const revid = data['revid'][idx];
    const url = data['url'][idx];
    const title = data['title'][idx];
    const text = data['text'][idx];
    rows.push(
      createData(
        id,
        revid,
        url,
        title,
        text
      )
    );
  });

  // TODO: Consider lazy loading... for now let's take 500 elements.
  rows = rows.slice(0, 500);

  return rows;
}

const useStyles = makeStyles({
  root: {
    width: '100%',
  },
  container: {
    maxHeight: 440,
  },
});

export function QueryResultTable({ data, queryTime }) {
  const classes = useStyles();
  const [page, setPage] = React.useState(0);
  const [rowsPerPage, setRowsPerPage] = React.useState(10);

  const handleChangePage = (event, newPage) => {
    setPage(newPage);
  };

  const handleChangeRowsPerPage = (event) => {
    setRowsPerPage(+event.target.value);
    setPage(0);
  };

  const rows = formatData(data);

  return (
    <Paper className={classes.root}>
      <Typography style={{ marginLeft: 5 }} variant="h6" id="tableTitle" component="div">
        Query Time: {Math.round(queryTime)} ms
      </Typography>
      <TableContainer className={classes.container}>
        <Table stickyHeader aria-label="sticky table">
          <TableHead>
            <TableRow>
              {columns.map((column) => (
                <TableCell
                  key={column.id}
                  align={column.align}
                  style={{ minWidth: column.minWidth }}
                >
                  {column.label}
                </TableCell>
              ))}
            </TableRow>
          </TableHead>
          <TableBody>
            {rows.slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage).map((row) => {
              return (
                <TableRow hover role="checkbox" tabIndex={-1} key={row.code}>
                  {columns.map((column) => {
                    const value = row[column.id];
                    return (
                      <TableCell key={column.id} align={column.align}>
                        {column.format && typeof value === 'number' ? column.format(value) : value}
                      </TableCell>
                    );
                  })}
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </TableContainer>
      <TablePagination
        rowsPerPageOptions={[10, 25, 100, 500]}
        component="div"
        count={rows.length}
        rowsPerPage={rowsPerPage}
        page={page}
        onPageChange={handleChangePage}
        onRowsPerPageChange={handleChangeRowsPerPage}
      />
    </Paper>
  );
}
