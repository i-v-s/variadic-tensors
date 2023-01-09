cmake_minimum_required(VERSION 3.5)

project(variadic-tensors LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# add_compile_options(-fsanitize=address)
# add_link_options(-fsanitize=address)


# Library

find_package(CUDAToolkit REQUIRED)

add_library(${PROJECT_NAME})

target_sources(${PROJECT_NAME} PRIVATE
    src/axis.h
    src/utils.h
    src/shape.h
    src/strides.h
    src/slice.h
    src/core.h     src/core.cpp
    src/cuda.h     src/cuda.cpp
)

target_include_directories(${PROJECT_NAME}
    PRIVATE
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/>"
    PUBLIC
        "$<INSTALL_INTERFACE:include/>")

target_link_libraries(${PROJECT_NAME} PUBLIC
    tbb
    CUDA::cudart CUDA::nppc CUDA::nppig)


# Application

find_package(OpenCV REQUIRED)

add_executable(${PROJECT_NAME}-app main.cpp)
target_link_libraries(${PROJECT_NAME}-app
    ${PROJECT_NAME}
    opencv_core opencv_highgui opencv_imgproc
)
