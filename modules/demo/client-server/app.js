var createError = require('http-errors');
var express     = require('express');
var path        = require('path');
var logger      = require('morgan');
var cors        = require('cors')

const app = require('express')();
app.io    = require('socket.io')();

var indexRouter = require('./routes/index');
var cudfRouter  = require('./routes/cudf')(app.io);

app.use(logger('dev'));
app.set('views', path.join(__dirname, 'views'));
app.set('view engine', 'jade');

app.use(express.json());
app.use(express.urlencoded({extended: false}));

app.use(express.static(path.join(__dirname, 'public'), {extensions: ['html']}));

app.use('/', indexRouter);
app.use('/cudf', cudfRouter);

// catch 404 and forward to error handler
app.use(function(req, res, next) { next(createError(404)); });

// error handler
app.use(function(err, req, res, next) {
  // set locals, only providing error in development
  res.locals.message = err.message;
  res.locals.error   = req.app.get('env') === 'development' ? err : {};

  // render the error page
  res.status(err.status || 500);
  res.render('error');
});

process.on('SIGINT', () => process.exit(1));

console.log('dashboard demos running on: http://localhost:3000')
module.exports = app;
