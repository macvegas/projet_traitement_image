cmake_minimum_required(VERSION 3.6)
project(opencv_test)
find_package(OpenCV REQUIRED)

# Gets all source files
file(GLOB_RECURSE MY_SOURCES src/*)

# Gets all header files
file(GLOB_RECURSE MY_HEADERS include/*)
# Adds OpenCV header files
set( MY_HEADERS ${MY_HEADERS} ${OpenCV_INCLUDE_DIRS})

message("OpenCV includes" ${OpenCV_INCLUDE_DIRS})
message("OpenCV libs" ${OpenCV_LIBS})

include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/include ${OpenCV_INCLUDE_DIRS})
#link_directories( ${CMAKE_BINARY_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
add_executable(my_project ${MY_SOURCES} ${MY_HEADERS} )

target_link_libraries(my_project ${OpenCV_LIBS} )
