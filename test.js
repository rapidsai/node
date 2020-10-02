const cudf = require('@nvidia/cudf')
const rmm = require('@nvidia/rmm')

let r = new rmm.DeviceBuffer(10,10)

let c = new cudf.Column('int32', 10, r)

console.log(c.size(), c.type());

console.log(c.has_nulls(), c.null_count());

console.log(c.nullable());
console.log(c.size());









