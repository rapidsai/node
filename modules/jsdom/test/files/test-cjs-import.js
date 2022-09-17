const testCJSModule = require('./test-cjs-module');

module.exports = {
  importedModuleSharesGlobalsWithThisModule: Object.aGlobalField === testCJSModule.aGlobalField,
};
