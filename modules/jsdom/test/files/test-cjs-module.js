Object.defineProperty(Object, 'aGlobalField', {value: 10});

exports.foo          = 'foo';
exports.aGlobalField = Object.aGlobalField;
exports.setFooToBar = () => { exports.foo = 'bar'; };
