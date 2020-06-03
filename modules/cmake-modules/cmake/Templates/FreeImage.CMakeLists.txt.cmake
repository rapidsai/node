cmake_minimum_required(VERSION 3.12)

include(ExternalProject)

ExternalProject_Add(FreeImage
                    URL                     https://svwh.dl.sourceforge.net/project/freeimage/Source%20Distribution/3.18.0/FreeImage3180.zip
                    DOWNLOAD_NO_PROGRESS    TRUE
                    SOURCE_DIR              "${FREEIMAGE_ROOT}/freeimage"
                    BINARY_DIR              "${FREEIMAGE_ROOT}/build"
                    INSTALL_DIR             "${CMAKE_CURRENT_BINARY_DIR}"
                    CMAKE_ARGS              ${FREEIMAGE_CMAKE_ARGS} -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}
                    PATCH_COMMAND           ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_LIST_DIR}/../Templates/FreeImage/CMakeLists.txt <SOURCE_DIR>/CMakeLists.txt)
