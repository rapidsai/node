add_definitions("-DNAPI_EXPERIMENTAL")

execute_process(COMMAND node -p "require('node-addon-api').include"
                WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                OUTPUT_VARIABLE NODE_ADDON_API_DIR)

string(REPLACE "\n" "" NODE_ADDON_API_DIR "${NODE_ADDON_API_DIR}")
string(REPLACE "\"" "" NODE_ADDON_API_DIR "${NODE_ADDON_API_DIR}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNODE_ADDON_API_DISABLE_DEPRECATED")

include_directories("${CMAKE_JS_INC}"
                    "${PROJECT_NAME}"
            PRIVATE "${NODE_ADDON_API_DIR}")
