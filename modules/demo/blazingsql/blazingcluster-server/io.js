const {SQLContext} = require('@rapidsai/blazingsql');

const bc = new SQLContext();
bc.createCSVTable('test_table', [`${__dirname}/wikipedia_pages.csv`]);

const result = bc.sql("SELECT * FROM test_table").result();

console.log(result.names);

result.names.forEach((n) => {
    console.log([...result.get(n)]);
});
