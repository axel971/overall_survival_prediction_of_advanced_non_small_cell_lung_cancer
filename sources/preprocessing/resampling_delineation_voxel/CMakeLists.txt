

cmake_minimum_required(VERSION 2.5)

project(resampling_delineation_voxel)

# Find ITK
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

add_executable(resampling_delineation_voxel ./source/resample_delineation_voxel.cpp)

target_link_libraries(resampling_delineation_voxel ${ITK_LIBRARIES})
