const {
  defineConfig,
} = require('eslint/config');

const globals          = require('globals');
const typescriptEslint = require('@typescript-eslint/eslint-plugin');
const tsParser         = require('@typescript-eslint/parser');
const js               = require('@eslint/js');

const {
  FlatCompat,
} = require('@eslint/eslintrc');

const compat = new FlatCompat(
  {baseDirectory: __dirname, recommendedConfig: js.configs.recommended, allConfig: js.configs.all});

module.exports = defineConfig([
  {
    languageOptions : {
      globals : {
        ...globals.browser,
        ...globals.node,
      },

      parser : tsParser,
      'sourceType' : 'module',

      parserOptions : {
        'project' : ['tsconfig.json'],
      },
    },

    plugins : {
      '@typescript-eslint' : typescriptEslint,
    },

    extends : compat.extends(
                              'eslint:recommended',
                              'plugin:@typescript-eslint/recommended',
                              'plugin:@typescript-eslint/recommended-requiring-type-checking',
                              ),

    'rules' : {
      'semi' : ['error', 'always'],
      'no-useless-assignment' : 'off',
      '@typescript-eslint/unbound-method' : 'off',
      '@typescript-eslint/ban-ts-comment' : 'off',
      '@typescript-eslint/no-unsafe-call' : 'off',
      '@typescript-eslint/no-var-requires' : 'off',
      '@typescript-eslint/no-explicit-any' : 'off',
      '@typescript-eslint/no-unsafe-return' : 'off',
      '@typescript-eslint/no-empty-function' : 'off',
      '@typescript-eslint/no-require-imports' : 'off',
      '@typescript-eslint/no-unsafe-argument' : 'off',
      '@typescript-eslint/no-unsafe-assignment' : 'off',
      '@typescript-eslint/no-unsafe-member-access' : 'off',
      '@typescript-eslint/no-unused-expressions' : 'off',
      '@typescript-eslint/explicit-module-boundary-types' : 'off',
      '@typescript-eslint/no-redeclare' : 'off',

      '@typescript-eslint/no-use-before-define' :
                                                [
                                                  'error',
                                                  {
                                                    'functions' : false,
                                                    'classes' : false,
                                                    'variables' : false,
                                                  }
                                                ],
    },
  },
  {
    files : ['test/**/*.ts'],

    'rules' : {
      'no-debugger' : 'off',
    },
  }
]);
