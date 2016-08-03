cmake_minimum_required(VERSION 2.8)
find_package(Boost 1.50.0 COMPONENTS program_options REQUIRED)
find_package(VTK)
if (VTK_FOUND)
    add_definitions(-DHAVE_VTK)
endif(VTK_FOUND)

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-Wno-deprecated -std=c++11 -fopenmp")
set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-undefined")

include_directories(${VTK_INCLUDE_DIRS} ${Boost_INCLUDE_DIR} Aboria/src)

add_executable(soft_spheres soft_spheres.cpp)
target_link_libraries(soft_spheres ${VTK_LIBRARIES} ${Boost_LIBRARIES})

