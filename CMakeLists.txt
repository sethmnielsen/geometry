cmake_minimum_required (VERSION 2.8.11)
project (geometry)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
find_package(GTest REQUIRED)
find_package(Eigen3 REQUIRED)

add_library(geometry INTERFACE)
target_include_directories(geometry INTERFACE include)

if (${GTEST_FOUND})
	include_directories(include ${GTEST_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS})
	add_executable(geometry_test src/test.cpp)
	target_link_libraries(geometry_test ${GTEST_LIBRARIES} gtest gtest_main pthread)
endif(${GTEST_FOUND})
