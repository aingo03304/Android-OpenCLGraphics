cmake_minimum_required(VERSION 3.4.1)

add_subdirectory(libopencl-stub)
add_subdirectory(csparse)

find_library(
        log-lib

        log)

find_library(
        android-lib

        android)

add_library( # Sets the name of the library.
        native-lib

        # Sets the library as a shared library.
        SHARED

        # Provides a relative path to your source file(s).
        native-lib.cpp
        )

target_include_directories(
        native-lib
        PRIVATE
        libopencl-stub/include/
        libopencl-stub/include/CL/
        csparse/include/
        kernels/
)

target_link_libraries( # Specifies the target library.
        native-lib
        OpenCL
        Csparse
        dl
        log
        mediandk
        
        # Links the target library to the log library
        # included in the NDK.
        ${log-lib}

        ${android-lib})
