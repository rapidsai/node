import {
  Bool8,
  Column,
  DataFrame,
  DataType,
  Float32,
  Float32Series,
  Float64,
  Float64Series,
  Int16,
  Int16Series,
  Int32,
  Int32Series,
  Int64,
  Int64Series,
  Int8,
  Int8Series,
  Series,
  SeriesMap,
  StringSeries,
  TimestampDay,
  TimestampMicrosecond,
  TimestampMillisecond,
  TimestampNanosecond,
  TimestampSecond,
  Uint16,
  Uint16Series,
  Uint32,
  Uint32Series,
  Uint64,
  Uint64Series,
  Uint8,
  Uint8Series,
  Utf8String
} from '@rapidsai/cudf';

/* TODO: How do I apply a list of dtypes?
 */
function json_aos_to_dataframe(
  str: StringSeries, columns: ReadonlyArray<string>, _: ReadonlyArray<DataType>): DataFrame {
  const arr = {} as SeriesMap;
  columns.forEach((col, ix) => {
    const no_open_list = str.split('[\n').gather([1], false);
    const tokenized    = no_open_list.split('},');
    console.log(tokenized.toArray());
    const parse_result = tokenized._col.getJSONObject('.' + columns[ix]);
    arr[col]           = Series.new(parse_result);
    console.log(Series.new(parse_result).toArray());
  });
  const result = new DataFrame(arr);
  return result;
}
/* TODO: How do I apply a list of dtypes?
 */
function json_aoa_to_dataframe(str: StringSeries, dtypes: ReadonlyArray<DataType>): DataFrame {
  const arr = {} as SeriesMap;
  dtypes.forEach((_, ix) => {
    const no_open_list = str.split('[\n').gather([1], false);
    const tokenized    = no_open_list.split('],');
    const get_ix       = `[${ix}]`;
    const parse_result = tokenized._col.getJSONObject(get_ix);
    arr[ix]            = Series.new(parse_result);
  });
  const result = new DataFrame(arr);
  return result;
}

module.exports = async function(fastify: any, opts: any) {
  fastify.get('/', async function(request: any, reply: any) {
    const x = Series.new([1, 2, 3]);
    return { root: true, data: x.toArray() }
  });
  fastify.get('/hello', async function(request: any, reply: any) { return 'hello'; });
  fastify.route({
    method: 'GET',
    url: '/load-graphology-json',
    schema: {
      querystring: {filename: {type: 'string'}},

      response: {
        200: {
          type: 'object',
          properties:
            {success: {type: 'boolean'}, message: {type: 'string'}, params: {type: 'string'}}
        }
      }
    },
    handler: async (request: any, reply: any) => {
      // load the file via read_text
      // parse it with json etc
      // store the results in a global cache
      // return that it is done
      return { 'params': request.query.filename, success: false, message: 'Not implemented.' }
    }
  });
}
