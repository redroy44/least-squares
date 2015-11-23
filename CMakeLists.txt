cmake_minimum_required(VERSION 3.0)

project (least-squares)

set(Boost_USE_STATIC_LIBS OFF) 
set(Boost_USE_MULTITHREADED ON)  
set(Boost_USE_STATIC_RUNTIME OFF) 
find_package(Boost REQUIRED)
find_package(Armadillo REQUIRED)

enable_testing()
find_package(GTest REQUIRED)

MESSAGE(STATUS ${Boost_circular_buffer_LIBRARY})

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG -O0")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include
                    ${ARMADILLO_INCLUDE_DIR}
                    ${Boost_INCLUDE_DIRS}
                    ${GTEST_INCLUDE_DIRS})

add_library(ls SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/ls.cpp       
                      ${CMAKE_CURRENT_SOURCE_DIR}/src/rls.cpp)

set(CMAKE_CXX_FLAGS "-g -Wall -std=c++14")

message(STATUS "Default flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "Release flags: ${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "Debug flags: ${CMAKE_CXX_FLAGS_DEBUG}")

add_executable(runTest test/test.cpp)

target_link_libraries(ls ${ARMADILLO_LIBRARIES})
target_link_libraries(runTest ls ${ARMADILLO_LIBRARIES} ${GTEST_BOTH_LIBRARIES})

add_test(AllTests runTest)
