module.exports = {
  // The root directory that Jest should scan for tests and modules
  rootDir: './',

  // The glob patterns Jest should use to find test files
  testMatch: ['**/__tests__/**/*.js?(x)', '**/?(*.)+(spec|test).js?(x)'],

  // The environment that Jest should use to run tests
  testEnvironment: 'jsdom',

  // The module file extensions that Jest should look for
  moduleFileExtensions: ['js', 'jsx'],

  // A list of paths to directories that Jest should use to search for modules
  moduleDirectories: ['node_modules'],

  // A list of paths to modules that Jest should use to transform code
  transform: {
    '^.+\\.jsx?$': 'babel-jest',
  },

  // A list of paths to modules that Jest should not transform
  transformIgnorePatterns: ['<rootDir>/node_modules/'],
};

