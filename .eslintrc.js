module.exports = {
  'env': {
    'browser': true,
    'es6': true,
    'node': true,
  },
  'plugins': ['@typescript-eslint'],
  'parser': '@typescript-eslint/parser',
  'parserOptions': {
    'project': [
      'tsconfig.json'
    ],
    'sourceType': 'module'
  },
  'extends': [
    'eslint:recommended',
    'plugin:@typescript-eslint/recommended',
    'plugin:@typescript-eslint/recommended-requiring-type-checking'
  ],
  'rules': {
    // "prefer-const": "off",
    // "prefer-rest-params": "off",
    'semi': ['error', 'always'],
    '@typescript-eslint/unbound-method': 'off',
    '@typescript-eslint/ban-ts-comment': 'off',
    '@typescript-eslint/no-unsafe-call': 'off',
    '@typescript-eslint/no-var-requires': 'off',
    '@typescript-eslint/no-explicit-any': 'off',
    '@typescript-eslint/no-unsafe-return': 'off',
    '@typescript-eslint/no-empty-function': 'off',
    '@typescript-eslint/no-require-imports': 'off',
    '@typescript-eslint/no-unsafe-argument': 'off',
    '@typescript-eslint/no-unsafe-assignment': 'off',
    // "@typescript-eslint/no-non-null-assertion": "off",
    '@typescript-eslint/no-unsafe-member-access': 'off',
    '@typescript-eslint/explicit-module-boundary-types': 'off',

    '@typescript-eslint/no-redeclare': [
      'error',
      {'ignoreDeclarationMerge': true},
    ],
    '@typescript-eslint/no-use-before-define': [
      'error',
      {'functions': false, 'classes': false, 'variables': false},
    ],
  },
  'overrides': [
    {'files': ['test/**/*.ts'], 'rules': {'no-debugger': 'off'}},
  ]
};
