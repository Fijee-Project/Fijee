message(STATUS "Checking for EIGEN3")


if(NOT ${EIGEN_DIR})
  set(EIGEN_DIR "/FIJEE-AUX")
endif()

set(EIGEN_INCLUDE_DIR "/FIJEE-AUX/include/eigen3/")


set(EIGEN_FOUND 1)

