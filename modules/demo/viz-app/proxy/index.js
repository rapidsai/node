var proxy = require('redbird')({port: 3000, bunyan: false});
proxy.register('http://localhost:3000', 'http://localhost:3001/');
proxy.register('http://localhost:3000/api', 'http://localhost:8080/');
