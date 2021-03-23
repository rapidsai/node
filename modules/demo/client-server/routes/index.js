var express = require('express');
var router  = express.Router();
var path    = require('path');

/* GET home page. */
router.get('/', function(req, res, next) { res.redirect('/front-end/app.html') });

module.exports = router;
