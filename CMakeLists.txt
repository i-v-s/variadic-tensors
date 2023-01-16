cmake_minimum_required(VERSION 3.5)

project(variadic-tensors LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# add_compile_options(-fsanitize=address)
# add_link_options(-fsanitize=address)


# Library

find_package(CUDAToolkit REQUIRED)
find_package(OpenCV OPTIONAL_COMPONENTS core)

add_library(${PROJECT_NAME})

target_sources(${PROJECT_NAME} PRIVATE
    variadic_tensors/axis.h
    variadic_tensors/utils.h
    variadic_tensors/shape.h
    variadic_tensors/strides.h
    variadic_tensors/slice.h
    variadic_tensors/buffers.h
    variadic_tensors/actions.h
    variadic_tensors/core.h     src/core.cpp
    variadic_tensors/cuda.h     src/cuda.cpp
)

target_include_directories(${PROJECT_NAME}
    PRIVATE
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/variadic_tensors/>"
    PUBLIC
        "$<INSTALL_INTERFACE:include/>")

target_link_libraries(${PROJECT_NAME} PUBLIC
    tbb
    CUDA::cudart CUDA::nppc CUDA::nppig)

if(OpenCV_core_FOUND)
    target_sources(${PROJECT_NAME} PRIVATE variadic_tensors/ocv.h)
    target_link_libraries(${PROJECT_NAME} PUBLIC opencv_core)
endif()


# Install library

include(GNUInstallDirs)

install(
    TARGETS ${PROJECT_NAME}
    EXPORT ${PROJECT_NAME}-config
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )

install(
    EXPORT ${PROJECT_NAME}-config
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})

install(DIRECTORY variadic_tensors
        DESTINATION include)


# Build .deb package

set(CPACK_GENERATOR "DEB")
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Igor Smirnov")
# set(CPACK_DEBIAN_PACKAGE_DEPENDS "")
include(CPack)


# Application

find_package(OpenCV REQUIRED)

add_executable(${PROJECT_NAME}-app main.cpp)
target_link_libraries(${PROJECT_NAME}-app
    ${PROJECT_NAME}
    opencv_core opencv_highgui opencv_imgproc
)
