var createError = require('http-errors');
var express = require('express');
var Path = require('path');
var logger = require('morgan');

const app = require('express')();
app.io = require('socket.io')();

app
  .set('views', Path.join(__dirname, 'views'))
  .set('view engine', 'jade')
  .use(logger('dev'))
  .use(require('cors')())
  .use(require('cookie-parser')('keyboard cat'))
  .use(express.json())
  .use(express.urlencoded({ extended: true }))
  .use(express.static(Path.join(__dirname, 'public'), { extensions: ['html'] }))
  .use('/', require('./routes/index'))
  // Add router to handle HTTP requests to `/uber` datasets
  .use('/uber', require('./routes/uber')())
  .use('/cudf', require('./routes/cudf')(app.io))
  // catch 404 and forward to error handler
  .use(function (req, res, next) { next(createError(404)); })
  // error handler
  .use(function (err, req, res, next) {
    // set locals, only providing error in development
    res.locals.message = err.message;
    res.locals.error = req.app.get('env') === 'development' ? err : {};
    // render the error page
    res.status(err.status || 500);
    res.render('error');
  });

['SIGTERM', 'SIGINT', 'SIGBREAK', 'SIGHUP'].forEach((signal) => {
  signal === 'SIGINT'
    ? process.on(signal, () => process.exit(0))
    : process.on(signal, () => process.exit(1))
});

console.log('dashboard demos running on: http://localhost:3000');

module.exports = app;
