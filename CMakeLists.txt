cmake_minimum_required(VERSION 3.0)

project (kalman)

find_package(Armadillo REQUIRED)

set(CMAKE_BUILD_TYPE Release)

SET(LS_LIBRARY ${CMAKE_CURRENT_SOURCE_DIR}/least-squares)

include_directories(${LS_LIBRARY}/include)
add_library(ls SHARED ${LS_LIBRARY}/src/ls.cpp ${LS_LIBRARY}/src/rls.cpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -std=c++11 -O0")

add_executable(main main.cpp)

target_link_libraries(ls armadillo)
target_link_libraries(main ls)
