import {BlazingContext} from '@rapidsai/blazingsql';
import {DataFrame, Series} from '@rapidsai/cudf';

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

test('base case', () => {
  const a  = Series.new([6, 9, 1, 6, 2]);
  const b  = Series.new([7, 2, 7, 1, 2]);
  const df = new DataFrame({'a': a, 'b': b});

  const bc = new BlazingContext();
  bc.createTable('test_table', df);

  expect(bc.sql('SELECT a FROM test_table')).toStrictEqual(new DataFrame({a}));
});
