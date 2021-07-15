import {BlazingContext} from '@rapidsai/blazingsql';
import {DataFrame, Float64, Series, Utf8String} from '@rapidsai/cudf';

test('create and drop table', () => {
  const a  = Series.new([1, 2, 3]);
  const df = new DataFrame({'a': a});

  const bc = new BlazingContext();
  bc.createTable('test_table', df);
  expect(bc.listTables().length).toEqual(1);

  bc.dropTable('test_table');
  expect(bc.listTables().length).toEqual(0);
});

test('list tables', () => {
  const a  = Series.new([1, 2, 3]);
  const df = new DataFrame({'a': a});

  const bc = new BlazingContext();
  bc.createTable('test_table', df);
  bc.createTable('test_table2', df);

  expect(bc.listTables()).toEqual(['test_table', 'test_table2']);
});

test('describe table', () => {
  const a  = Series.new([1, 2, 3]);
  const b  = Series.new(['foo', 'bar', 'foo']);
  const df = new DataFrame({'a': a, 'b': b});

  const bc = new BlazingContext();

  expect(() => bc.describeTable('nonexisting_table')).toThrow();

  bc.createTable('test_table', df);
  const tableDescription = bc.describeTable('test_table');
  expect(Object.keys(tableDescription)).toEqual(['a', 'b']);
  expect(Object.values(tableDescription)).toEqual([new Float64, new Utf8String]);
});

test('select a single column', () => {
  const a  = Series.new([6, 9, 1, 6, 2]);
  const b  = Series.new([7, 2, 7, 1, 2]);
  const df = new DataFrame({'a': a, 'b': b});

  const bc = new BlazingContext();
  bc.createTable('test_table', df);

  expect(bc.sql('SELECT a FROM test_table')).toStrictEqual(new DataFrame({a}));
});

test('select all columns', () => {
  const a  = Series.new([6, 9, 1, 6, 2]);
  const b  = Series.new([7, 2, 7, 1, 2]);
  const df = new DataFrame({'a': a, 'b': b});

  const bc = new BlazingContext();
  bc.createTable('test_table', df);

  expect(bc.sql('SELECT * FROM test_table')).toStrictEqual(new DataFrame({'a': a, 'b': b}));
});

test('union columns from two tables', () => {
  const a   = Series.new([1, 2, 3]);
  const df1 = new DataFrame({'a': a});
  const df2 = new DataFrame({'a': a});

  const bc = new BlazingContext();
  bc.createTable('t1', df1);
  bc.createTable('t2', df2);

  const result = new DataFrame({'a': Series.new([...a, ...a])});
  expect(bc.sql('SELECT a FROM t1 AS a UNION ALL SELECT a FROM t2')).toStrictEqual(result);
});

test('find all columns within a table that meet condition', () => {
  const key = Series.new(['a', 'b', 'c', 'd', 'e']);
  const val = Series.new([7.6, 2.9, 7.1, 1.6, 2.2]);
  const df  = new DataFrame({'key': key, 'val': val});

  const bc = new BlazingContext();
  bc.createTable('test_table', df);

  const result = new DataFrame({'key': Series.new(['a', 'b']), 'val': Series.new([7.6, 7.1])});
  expect(bc.sql('SELECT * FROM test_table WHERE val > 4')).toStrictEqual(result);
});
