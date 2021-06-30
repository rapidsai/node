import {BlazingContext} from '@rapidsai/blazingsql';
import {
  DataFrame,
  Series,
} from '@rapidsai/cudf';

test(`base case`, () => {
  const a      = Series.new([6, 9, 1, 6, 2]);
  const b      = Series.new([7, 2, 7, 1, 2]);
  const source = new DataFrame({'a': a, 'b': b});

  const bc = new BlazingContext();

  console.log(source);
  console.log(bc);
});
