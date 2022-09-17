import testESMModule from './test-esm-module';

export default {
  importedModuleSharesGlobalsWithThisModule: Object.aGlobalField === testESMModule.aGlobalField,
};
