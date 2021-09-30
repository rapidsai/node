import {DataFrame, Float64, Series, Utf8String} from '@rapidsai/cudf';
import {SQLContext} from '@rapidsai/sql';

test('create and drop table', () => {
  const a  = Series.new([1, 2, 3]);
  const df = new DataFrame({'a': a});

  const sqlContext = new SQLContext();
  sqlContext.createTable('test_table', df);
  expect(sqlContext.listTables().length).toEqual(1);

  sqlContext.dropTable('test_table');
  expect(sqlContext.listTables().length).toEqual(0);
});

test('list tables', () => {
  const a  = Series.new([1, 2, 3]);
  const df = new DataFrame({'a': a});

  const sqlContext = new SQLContext();
  sqlContext.createTable('test_table', df);
  sqlContext.createTable('test_table2', df);

  expect(sqlContext.listTables()).toEqual(['test_table', 'test_table2']);
});

test('describe table', () => {
  const a  = Series.new([1, 2, 3]);
  const b  = Series.new(['foo', 'bar', 'foo']);
  const df = new DataFrame({'a': a, 'b': b});

  const sqlContext = new SQLContext();

  // Empty map since table doesn't exist
  expect(sqlContext.describeTable('nonexisting_table').size).toEqual(0);

  sqlContext.createTable('test_table', df);
  const tableDescription = sqlContext.describeTable('test_table');
  expect([...tableDescription.keys()]).toEqual(['a', 'b']);
  expect([...tableDescription.values()]).toEqual([new Float64, new Utf8String]);
});

test('explain', () => {
  const key = Series.new(['a', 'b', 'c', 'd', 'e']);
  const val = Series.new([7.6, 2.9, 7.1, 1.6, 2.2]);
  const df  = new DataFrame({'key': key, 'val': val});

  const sqlContext = new SQLContext();
  sqlContext.createTable('test_table', df);

  const query = 'SELECT * FROM test_table WHERE val > 4';

  expect(sqlContext.explain(query))
    .toEqual(
      `LogicalProject(key=[$0], val=[$1])
  BindableTableScan(table=[[main, test_table]], filters=[[>($1, 4)]])
`);
  expect(sqlContext.explain(query, true))
    .toEqual(
      `LogicalProject(key=[$0], val=[$1])
  BindableTableScan(table=[[main, test_table]], filters=[[>($1, 4)]])

`);
});

test('select a single column', async () => {
  const a  = Series.new([6, 9, 1, 6, 2]);
  const b  = Series.new([7, 2, 7, 1, 2]);
  const df = new DataFrame({'a': a, 'b': b});

  const sqlContext = new SQLContext();
  sqlContext.createTable('test_table', df);

  await expect(sqlContext.sql('SELECT a FROM test_table').result())
    .resolves.toStrictEqual(new DataFrame({a}));
});

test('select all columns', async () => {
  const a  = Series.new([6, 9, 1, 6, 2]);
  const b  = Series.new([7, 2, 7, 1, 2]);
  const df = new DataFrame({'a': a, 'b': b});

  const sqlContext = new SQLContext();
  sqlContext.createTable('test_table', df);

  await expect(sqlContext.sql('SELECT * FROM test_table').result())
    .resolves.toStrictEqual(new DataFrame({'a': a, 'b': b}));
});

test('union columns from two tables', async () => {
  const a   = Series.new([1, 2, 3]);
  const df1 = new DataFrame({'a': a});
  const df2 = new DataFrame({'a': a});

  const sqlContext = new SQLContext();
  sqlContext.createTable('t1', df1);
  sqlContext.createTable('t2', df2);

  const result = new DataFrame({'a': Series.new([...a, ...a])});
  await expect(sqlContext.sql('SELECT a FROM t1 AS a UNION ALL SELECT a FROM t2').result())
    .resolves.toStrictEqual(result);
});

test('find all columns within a table that meet condition', async () => {
  const key = Series.new(['a', 'b', 'c', 'd', 'e']);
  const val = Series.new([7.6, 2.9, 7.1, 1.6, 2.2]);
  const df  = new DataFrame({'key': key, 'val': val});

  const sqlContext = new SQLContext();
  sqlContext.createTable('test_table', df);

  const result = new DataFrame({'key': Series.new(['a', 'b']), 'val': Series.new([7.6, 7.1])});
  await expect(sqlContext.sql('SELECT * FROM test_table WHERE val > 4').result())
    .resolves.toStrictEqual(result);
});

test('empty sql result', async () => {
  const key = Series.new(['a', 'b', 'c', 'd', 'e']);
  const val = Series.new([7.6, 2.9, 7.1, 1.6, 2.2]);
  const df  = new DataFrame({'key': key, 'val': val});

  const sqlContext = new SQLContext();
  sqlContext.createTable('test_table', df);

  const result = new DataFrame();
  // Query should be empty since BETWEEN values are reversed.
  await expect(sqlContext.sql('SELECT * FROM test_table WHERE val BETWEEN 10 AND 0').result())
    .resolves.toStrictEqual(result);
});
