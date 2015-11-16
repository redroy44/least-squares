cmake_minimum_required(VERSION 3.0)

project (least-squares)

find_package(Armadillo REQUIRED)
set(Boost_USE_STATIC_LIBS OFF) 
set(Boost_USE_MULTITHREADED ON)  
set(Boost_USE_STATIC_RUNTIME OFF) 
find_package(Boost REQUIRED)

MESSAGE(STATUS ${Boost_circular_buffer_LIBRARY})

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG -O0")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include
                    ${ARMADILLO_INCLUDE_DIR}
                    ${Boost_INCLUDE_DIRS})

add_library(ls SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/ls.cpp       
                      ${CMAKE_CURRENT_SOURCE_DIR}/src/rls.cpp)

set(CMAKE_CXX_FLAGS "-g -Wall -std=c++11")

message(STATUS "Default flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "Release flags: ${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "Debug flags: ${CMAKE_CXX_FLAGS_DEBUG}")

add_executable(main main.cpp)

target_link_libraries(ls ${ARMADILLO_LIBRARIES})
target_link_libraries(main ls ${ARMADILLO_LIBRARIES})
