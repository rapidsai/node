import {BlazingCluster} from '@rapidsai/blazingsql';
import {DataFrame, Series} from '@rapidsai/cudf';

const delay = (ms: number) => new Promise(res => setTimeout(res, ms));

test('create and drop table', async () => {
  const a  = Series.new([1, 2, 3]);
  const df = new DataFrame({'a': a});

  const bc = new BlazingCluster();

  await delay(2000);
});
