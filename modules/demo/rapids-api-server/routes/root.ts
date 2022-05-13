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
import {fstat} from 'fs';

const Fs   = require('fs');
const Path = require('path');

/* TODO: How do I apply a list of dtypes?
 */
function json_aos_to_dataframe(
  str: StringSeries, columns: ReadonlyArray<string>, _: ReadonlyArray<DataType>): DataFrame {
  const arr = {} as SeriesMap;
  columns.forEach((col, ix) => {
    const no_open_list = str.split('[\n').gather([1], false);
    const tokenized    = no_open_list.split('},');
    const parse_result = tokenized._col.getJSONObject('.' + columns[ix]);
    arr[col]           = Series.new(parse_result);
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
  fastify.get('/graphology', async function(request: any, reply: any) {
    return {
      graphology: {
        description: 'The graphology api provides GPU acceleration of graphology datasets.',
        schema: {
          read_json: {
            filename: 'A URI to a graphology json dataset file.',
            returns: 'Result OK/Not Found/Fail'

          },
          list_tables:
            {returns: 'An object containing graphology related datasets resident on GPU memory.'},

          ':table': {
            ':column': 'The name of the column you want to request.',
            returns: 'An arrow buffer of the column contents.'
          }
        }
      }
    }
  });
  fastify.route({
    method: 'GET',
    url: '/graphology/read_json',
    schema: {
      querystring: {filename: {type: 'string'}, 'rootkeys': {type: 'array'}},

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
      // is the file local or remote?
      let message: string = 'Unknown error';
      let result = {'params': JSON.stringify(request.query), success: false, message: message};

      console.log(result);
      if (request.query.filename.search('http') != -1) {
        message = 'Remote files not supported yet.'
        console.log(result);
      } else {
        message = 'File is not remote';
        // does the file exist?
        const path = Path.join(__dirname, request.query.filename);
        try {
          Fs.stat(path, (err: any, stats: any) => {
            if (err) {
              message        = 'File does not exist at ' + path;
              result.message = message;
              reply.code(200).send(result);
            } else {
              // did the file read?
              message       = 'File is available';
              const dataset = StringSeries.read_text(path, '');
              message       = 'File could not be read';
              request.query.rootkeys.reverse().forEach((element: ' string') => {
                let split       = dataset.split('"tags":');
                const ttags     = split.gather([1], false);
                let rest        = split.gather([0], false);
                split           = rest.split('"clusters":');
                const tclusters = split.gather([1], false);
                rest            = split.gather([0], false);
                split           = rest.split('"edges":');
                const tedges    = split.gather([1], false);
                rest            = split.gather([0], false);
                split           = rest.split('"nodes":');
                const tnodes    = split.gather([1], false);
                const tags =
                  json_aos_to_dataframe(ttags, ['key', 'image'], [new Utf8String, new Utf8String]);
                const clusters = json_aos_to_dataframe(tclusters,
                                                       ['key', 'color', 'clusterLabel'],
                                                       [new Int32, new Utf8String, new Utf8String]);
                const nodes    = json_aos_to_dataframe(
                  tnodes, ['key', 'label', 'tag', 'URL', 'cluster', 'x', 'y', 'score'], [
                    new Utf8String,
                    new Utf8String,
                    new Utf8String,
                    new Utf8String,
                    new Int32,
                    new Float64,
                    new Float64,
                    new Int32
                  ]);
                const edges = json_aoa_to_dataframe(tedges, [new Utf8String, new Utf8String]);
              });
              result.success = true;
              message        = 'Successfully parsed json file onto GPU.';
              result.message = message;
              reply.code(200).send(result);
            }
          });
        } catch {
          message        = 'Exception reading file.';
          result.message = message;
          reply.code(200).send(result);
        };
      }
    }
  });
}
