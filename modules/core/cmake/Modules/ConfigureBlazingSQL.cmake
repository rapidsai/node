#=============================================================================
# Copyright (c) 2021, NVIDIA CORPORATION.
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

    _clean_build_dirs_if_not_fully_built(blazingsql-io libblazingsql-io.so)

    _set_package_dir_if_exists(blazingsql-io blazingsql-io)

    if(NOT TARGET blazingdb::blazingsql-io)
        _fix_rapids_cmake_dir()
        # _get_major_minor_version(${VERSION} MAJOR_AND_MINOR)
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
        _fix_rapids_cmake_dir()
    endif()

    # Make sure consumers of our libs can see blazingdb::blazingsql-io
    _fix_cmake_global_defaults(blazingdb::blazingsql-io)

    _clean_build_dirs_if_not_fully_built(blazingsql-engine libblazingsql-engine.so)

    _set_package_dir_if_exists(blazingsql-engine blazingsql-engine)

    if(NOT TARGET blazingdb::blazingsql-engine)
        _fix_rapids_cmake_dir()
        # _get_major_minor_version(${VERSION} MAJOR_AND_MINOR)
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
        _fix_rapids_cmake_dir()
    endif()

    # Make sure consumers of our libs can see blazingdb::blazingsql-engine
    _fix_cmake_global_defaults(blazingdb::blazingsql-engine)

    if (blazingsql-engine_ADDED)
        execute_process(COMMAND mvn clean install --quiet -f pom.xml -Dmaven.test.skip=true
                                -Dmaven.repo.local=${blazingsql-engine_BINARY_DIR}/blazing-protocol-mvn/
                        WORKING_DIRECTORY "${blazingsql-engine_SOURCE_DIR}/algebra")
    configure_file("${blazingsql-engine_SOURCE_DIR}/algebra/blazingdb-calcite-application/target/BlazingCalcite.jar"
                   "${blazingsql-engine_BINARY_DIR}/blazingsql-algebra.jar"
                   COPYONLY)

    configure_file("${blazingsql-engine_SOURCE_DIR}/algebra/blazingdb-calcite-core/target/blazingdb-calcite-core.jar"
                   "${blazingsql-engine_BINARY_DIR}/blazingsql-algebra-core.jar"
                   COPYONLY)

    endif()

    if(NOT blazingsql-engine_BINARY_DIR)
        set(blazingsql-engine_BINARY_DIR "${CPM_BINARY_CACHE}/blazingsql-engine-build")
        if(DEFINED ENV{NODE_RAPIDS_USE_LOCAL_DEPS_BUILD_DIRS})
            set(blazingsql-engine_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/_deps/blazingsql-engine-build")
        endif()
    endif()

    configure_file("${blazingsql-engine_BINARY_DIR}/blazingsql-algebra.jar"
                   "${CMAKE_CURRENT_BINARY_DIR}/blazingsql-algebra.jar"
                   COPYONLY)

    configure_file("${blazingsql-engine_BINARY_DIR}/blazingsql-algebra-core.jar"
                   "${CMAKE_CURRENT_BINARY_DIR}/blazingsql-algebra-core.jar"
                   COPYONLY)

endfunction()

find_and_configure_blazingsql(${BLAZINGSQL_VERSION})
