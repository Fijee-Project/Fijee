#
# Find the EIGEN includes and libraries
#
# EIGEN is a library that implements a variety of algorithms for
# linear algebra, sparse matrices. 
# It can be found at:
#
# EIGEN_INCLUDE_DIR - where to find metis.h
# EIGEN_LIBRARIES   - List of fully qualified libraries to link against.
# EIGEN_FOUND       - Do not attempt to use if "no" or undefined.

find_path( EIGEN_INCLUDE_DIR Eigen
  ${EIGEN_DIR}/include/eigen3
  /usr/include/eigen3
  /usr/include
  /opt/local/include
  /usr/local/include
  )

if(EIGEN_INCLUDE_DIR)
  set( EIGEN_FOUND "YES" )
else()
  set( EIGEN_FOUND "NO" )
  message(STATUS "Eigen not found")
endif( EIGEN_INCLUDE_DIR )

mark_as_advanced( EIGEN_INCLUDE_DIR EIGEN_LIBRARY )
