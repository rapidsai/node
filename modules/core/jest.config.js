// Copyright (c) 2020, NVIDIA CORPORATION.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

module.exports = {
    "cache": false,
    "verbose": false,
    "reporters": [
      "jest-silent-reporter"
    ],
    "testEnvironment": "node",
    "globals": {
      "ts-jest": {
        "diagnostics": false,
        "tsConfig": "test/tsconfig.json"
      }
    },
    "rootDir": "./",
    "roots": [
      "<rootDir>/test/"
    ],
    "moduleFileExtensions": [
      "js",
      "ts",
      "tsx"
    ],
    "coverageReporters": [
      "lcov"
    ],
    "coveragePathIgnorePatterns": [
      "test\\/.*\\.(ts|tsx|js)$",
      "/node_modules/"
    ],
    "transform": {
      "^.+\\.jsx?$": "ts-jest",
      "^.+\\.tsx?$": "ts-jest"
    },
    "transformIgnorePatterns": [
      "/build/js/*$",
      "/node_modules/(?!web-stream-tools).+\\.js$"
    ],
    "testRegex": "(.*(-|\\.)(test|spec)s?)\\.(ts|tsx|js)$",
    "preset": "ts-jest",
    "testMatch": null,
    "moduleNameMapper": {
        "^@nvidia\/rapids-core(.*)": "<rootDir>/src/$1",
    }
};
