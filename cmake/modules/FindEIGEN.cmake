message(STATUS "Checking for EIGEN3")


if(NOT ${EIGEN_DIR})
  set(EIGEN_DIR "/usr/include/eigen3")
endif()

set(EIGEN_INCLUDE_DIR "/usr/include/eigen3")


set(EIGEN_FOUND 1)

