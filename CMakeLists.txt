cmake_minimum_required(VERSION 2.6)
project(ex1)

include_directories(
        lib/
        common/
)

link_directories(
        ${PROJECT_SOURCE_DIR}/lib
        ${PROJECT_SOURCE_DIR}/lib/CL
)

IF (WIN32)
    SET(CL_LIBRARIES OpenCL.lib)
ELSE (WIN32)
    SET(CL_LIBRARIES OpenCL)
ENDIF(WIN32)

add_definitions(-DSRC_PATH=\"${CMAKE_SOURCE_DIR}/\")

add_executable(basics basics.cpp basics.cl common/clutils.cpp)
target_link_libraries(basics ${CL_LIBRARIES})
