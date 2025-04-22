cmake_minimum_required(VERSION 3.28)
project(connect4_eval)

set(CMAKE_CXX_STANDARD 23)

add_library(connect4_eval SHARED library.cpp)
add_executable(connect4_eval_test library.cpp)
target_link_libraries(connect4_eval PUBLIC -static-libgcc -static-libstdc++ -static)

# copy the shared library to the output directory
add_custom_command(TARGET connect4_eval POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:connect4_eval>
        ${CMAKE_CURRENT_SOURCE_DIR}/../
        COMMENT "Copying shared library to output directory"
)
