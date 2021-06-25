#=============================================================================
# Copyright (c) 2020-2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=============================================================================

function(find_and_configure_blazingsql VERSION)

    include(get_cpm)

    include(ConfigureCUDF)

    if(${VERSION} MATCHES [=[([0-9]+)\.([0-9]+)\.([0-9]+)]=])
        set(MAJOR_AND_MINOR "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}")
    else()
        set(MAJOR_AND_MINOR "${VERSION}")
    endif()

    _clean_build_dirs_if_not_fully_built(blazingsql-io libblazingsql-io.so)

    _set_package_dir_if_exists(blazingsql-io blazingsql-io)

    _fix_rapids_cmake_dir()

    if(NOT TARGET blazingdb::blazingsql-io)
        CPMFindPackage(NAME     blazingsql-io
            VERSION             ${VERSION}
            # GIT_REPOSITORY      https://github.com/rapidsai/cugraph.git
            # GIT_TAG             branch-${MAJOR_AND_MINOR}
            GIT_REPOSITORY      https://github.com/trxcllnt/blazingsql.git
            GIT_TAG             fea/rapids-cmake
            SOURCE_SUBDIR       io
            OPTIONS             "S3_SUPPORT OFF"
                                "GCS_SUPPORT OFF"
                                "BUILD_TESTS OFF"
                                "BUILD_BENCHMARKS OFF"
                                "BLAZINGSQL_IO_USE_ARROW_STATIC OFF"
                                "BLAZINGSQL_IO_BUILD_ARROW_PYTHON OFF"
        )
        # Make sure consumers of our libs can see blazingdb::blazingsql-io
        _fix_cmake_global_defaults(blazingdb::blazingsql-io)
    endif()

    _clean_build_dirs_if_not_fully_built(blazingsql-engine libblazingsql-engine.so)

    _set_package_dir_if_exists(blazingsql-engine blazingsql-engine)

    _fix_rapids_cmake_dir()

    if(NOT TARGET blazingdb::blazingsql-engine)
        CPMFindPackage(NAME     blazingsql-engine
            VERSION             ${VERSION}
            # GIT_REPOSITORY      https://github.com/rapidsai/cugraph.git
            # GIT_TAG             branch-${MAJOR_AND_MINOR}
            GIT_REPOSITORY      https://github.com/trxcllnt/blazingsql.git
            GIT_TAG             fea/rapids-cmake
            SOURCE_SUBDIR       engine
            OPTIONS             "S3_SUPPORT OFF"
                                "GCS_SUPPORT OFF"
                                "MYSQL_SUPPORT OFF"
                                "SQLITE_SUPPORT OFF"
                                "POSTGRESQL_SUPPORT OFF"
                                "CUDA_STATIC_RUNTIME ON"
                                "BUILD_TESTS OFF"
                                "BUILD_BENCHMARKS OFF"
                                "DISABLE_DEPRECATION_WARNING ON"
                                "BLAZINGSQL_IO_USE_ARROW_STATIC OFF"
                                "BLAZINGSQL_IO_BUILD_ARROW_PYTHON OFF"
                                "BLAZINGSQL_ENGINE_BUILD_ARROW_PYTHON OFF"
                                "BLAZINGSQL_ENGINE_WITH_PYTHON_ERRORS OFF"
        )
        # Make sure consumers of our libs can see blazingdb::blazingsql-engine
        _fix_cmake_global_defaults(blazingdb::blazingsql-engine)
    endif()

    _fix_rapids_cmake_dir()

    if (blazingsql-engine_ADDED)
        execute_process(COMMAND mvn clean install --quiet -f pom.xml -Dmaven.test.skip=true
                                -Dmaven.repo.local=${blazingsql-engine_BINARY_DIR}/blazing-protocol-mvn/
                        WORKING_DIRECTORY "${blazingsql-engine_SOURCE_DIR}/algebra"
                        OUTPUT_VARIABLE NODE_RAPIDS_CMAKE_MODULES_PATH
                        OUTPUT_STRIP_TRAILING_WHITESPACE)

        configure_file("${blazingsql-engine_SOURCE_DIR}/algebra/blazingdb-calcite-application/target/BlazingCalcite.jar"
                       "${CMAKE_CURRENT_BINARY_DIR}/blazingsql-algebra.jar"
                       COPYONLY)

        configure_file("${blazingsql-engine_SOURCE_DIR}/algebra/blazingdb-calcite-core/target/blazingdb-calcite-core.jar"
                       "${CMAKE_CURRENT_BINARY_DIR}/blazingsql-algebra-core.jar"
                       COPYONLY)

    endif()

endfunction()

find_and_configure_blazingsql(${BLAZINGSQL_VERSION})
