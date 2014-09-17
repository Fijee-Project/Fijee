# Barebones CMAKE script for fijee project
project( Fijee CXX C ) 

#
#
#include_directories( ${CMAKE_CURRENT_SOURCE_DIR} )

#
# Minimal version of CMake:
cmake_minimum_required( VERSION 2.6.0 )

#
# Some best practice versioning and soversioning for now
set( Fijee_NAME ${PROJECT_NAME} )
set( Fijee_VERSION_MAJOR "0" )
set( Fijee_VERSION_MINOR "1" )
set( Fijee_VERSION_PATCH "1" )
#
set(Fijee_VERSION "${Fijee_VERSION_MAJOR}.${Fijee_VERSION_MINOR}.${Fijee_VERSION_PATCH}")
#
set( LIBRARY_VERSION "0.0.1" )
set( LIBRARY_SOVERSION "0" )

#
# This needs to be changed to use cmake scripts to determine correct flag
# c standard and remove deprecation warnings (caused by vtk) that clutter compile output
add_definitions( -std=c++0x -Wno-deprecated )  


#
# Find required packages
#
message("\n== Find required packages ==\n")


# Tell cmake where to look for modules, modules are scripts that help us find required libraries
set(CMAKE_MODULE_PATH 
  "${CMAKE_SOURCE_DIR}/cmake/modules;${CMAKE_MODULE_PATH}"
)

# include(FijeeDefaults)
set(LIBS)

#set(DOLFIN_DIR "/usr/share/dolfin/cmake") # Dolfin installs UseDolfin.cmake scripts, dummy findDolfin script in cmake/modules doesn't look here properly yet

#
# FEniCS
find_package(DOLFIN REQUIRED CONFIG)
if(DOLFIN_FOUND)
  message("DOLFIN FOUND at ${DOLFIN_INCLUDE_DIRS} ${DOLFIN_3D_PARTY_LIBRARIES}")
  include(${DOLFIN_USE_FILE})
  list (APPEND LIBS ${DOLFIN_3RD_PARTY_LIBRARIES} ${DOLFIN_LIBRARIES})
  message (${LIBS})
#  add_definitions(${DOLFIN_CXX_DEFINITIONS})
#  include_directories(${DOLFIN_INCLUDE_DIRS})
#  list (APPEND LIBS ${DOLFIN_3D_PARTY_LIBRARIES})
endif()

#
# VTK
find_package(VTK REQUIRED NO_MODULE)
if(VTK_FOUND)
  include(${VTK_USE_FILE})
  list(APPEND LIBS ${VTK_LIBRARIES})
endif()


##
## BOOST
#find_package(Boost COMPONENTS mpi program_options filesystem thread system iostreams)
#if(Boost_FOUND)
#  include_directories(${Boost_INCLUDE_DIRS})
#  list(APPEND LIBS ${Boost_LIBRARIES})
#endif()
##
#add_definitions( -DBOOST_ALL_DYN_LINK)

#
# Eigen
find_package(EIGEN 3.0 REQUIRED)
#
if(EIGEN_FOUND)
  message(STATUS "eigen found @ ${EIGEN_DIR} ${EIGEN_INCLUDE_DIR}")
  include_directories(${EIGEN_INCLUDE_DIR})
endif()

#
# MPI (FEniCS)
find_package(MPI REQUIRED)
#
if(MPI_FOUND)
  message(STATUS "MPI found @ ${MPI_INCLUDE_PATH} : ")
  include_directories(${MPI_INCLUDE_PATH})
endif()

##
## NIFTI
#find_package(NIFTI REQUIRED)
##
#if(NIFTI_FOUND)
#  message(STATUS "FOUND NIFTI")
#  include_directories(${NIFTI_INCLUDE_DIR})
#endif()

#
# CGAL
find_package(CGAL 4.3 REQUIRED)
#
if(CGAL_FOUND)
  message( STATUS "CGAL FOUND ${CGAL_DIR}" )
  add_definitions( -DCGAL_EIGEN3_ENABLED )
  include( ${CGAL_USE_FILE} )
endif()

#
# Build 
#
message("\n== Generating build files ==\n")

#
# Env
set(Fijee_LIBRARY_DIR ${CMAKE_BINARY_DIR}/lib)
set(Fijee_INCLUDE_DIR ${CMAKE_BINARY_DIR}/include)
set(Fijee_BIN_DIR ${CMAKE_BINARY_DIR}/bin)
#
include_directories( SYSTEM )

#
#
set(Fijee_CXX_DEFINITIONS "")
set(Fijee_LINK_FLAGS "")
set(Fijee_TARGET_LINK_LIBRARIES "")


#
# Import all the source 
add_subdirectory(Fijee)


