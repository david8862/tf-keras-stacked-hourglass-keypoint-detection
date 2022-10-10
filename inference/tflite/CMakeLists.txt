cmake_minimum_required(VERSION 3.5)
project(hourglassKeypoint LANGUAGES CXX)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
SET(CMAKE_BUILD_TYPE "Release")

#### specify the build flag
SET(CMAKE_CXX_FLAGS  "-std=c++14 -O2")
SET(TARGET_PLAT "linux_x86_64" CACHE STRING INTERNAL)

set(HOURGLASS_KEYPOINT_SRC
        hourglassKeypoint.cpp)

#set(TF_ROOT_PATH /mnt/d/Downloads/tensorflow)
link_directories("${TF_ROOT_PATH}/tensorflow/lite/tools/make/gen/${TARGET_PLAT}/lib/")

add_executable(hourglassKeypoint ${HOURGLASS_KEYPOINT_SRC})

target_include_directories(hourglassKeypoint PRIVATE ${TF_ROOT_PATH} ${TF_ROOT_PATH}/tensorflow/lite/tools/make/downloads/flatbuffers/include)
target_link_libraries(hourglassKeypoint PRIVATE libtensorflow-lite.a -lstdc++ -lpthread -lm -ldl -lrt)
#target_link_libraries(hourglassKeypoint PRIVATE -ltensorflow-lite -lstdc++ -lpthread -lm -ldl -lrt)
