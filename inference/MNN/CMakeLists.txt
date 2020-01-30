cmake_minimum_required(VERSION 3.5)
project(hourglassKeypoint LANGUAGES CXX)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
SET(CMAKE_BUILD_TYPE "Release")

#### specify the compiler flag
SET(CMAKE_CXX_FLAGS  "-std=c++11 -O2")

set(HOURGLASS_KEYPOINT_SRC
        hourglassKeypoint.cpp)

#set(MNN_ROOT_PATH /mnt/d/Projects/MNN)

include_directories("${MNN_ROOT_PATH}/include/" "${MNN_ROOT_PATH}/3rd_party/imageHelper/")
link_directories("${MNN_ROOT_PATH}/build/")
add_executable(hourglassKeypoint ${HOURGLASS_KEYPOINT_SRC})
target_link_libraries(hourglassKeypoint -lMNN -lstdc++ -lpthread)
#target_link_libraries(hourglassKeypoint libMNN.a -Wl,--whole-archive -Wl,--no-whole-archive -lstdc++ -lpthread)
