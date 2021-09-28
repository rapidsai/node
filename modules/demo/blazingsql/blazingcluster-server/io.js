const {BlazingContext} = require('@rapidsai/blazingsql');

const bc = new BlazingContext();
bc.createTableCSV('test_table', [`${__dirname}/wikipedia_pages.csv`]);
