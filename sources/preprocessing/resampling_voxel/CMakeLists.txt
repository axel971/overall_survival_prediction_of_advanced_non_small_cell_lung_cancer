

cmake_minimum_required(VERSION 2.5)

project(resampling_voxel)

# Find ITK
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

add_executable(resampling_voxel ./source/resample_voxel.cpp)

target_link_libraries(resampling_voxel ${ITK_LIBRARIES})
