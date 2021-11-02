import {testMethod} from '@rapidsai/io';

test('quick import test', () => {
  const result = testMethod('test');
  expect(result).toEqual('test');
});
