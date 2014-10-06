cmake_minimum_required( VERSION 2.6.0 )

#
# Application's name 
PROJECT(Alpha_rhythm_electrodes)

#
# Project conf
find_package( Fijee REQUIRED )
#
include_directories( ${Fijee_INCLUDE_DIRS} )
link_directories( ${Fijee_LIBRARY_DIRS} )

#
# Other libraries
find_package( PkgConfig REQUIRED)
pkg_check_modules( gsl REQUIRED gsl )
pkg_check_modules( zlib REQUIRED zlib )

#
# Other definition
add_definitions( -O3 -std=c++0x -Wno-deprecated )  

#
# Build application
add_executable( Alpha_rhythm_electrodes MACOSX_BUNDLE Alpha_rhythm_electrodes )
#
target_link_libraries( Alpha_rhythm_electrodes 
  ${Fijee_LIBRARIES} 
  ${VTK_LIBRARIES}
  ${gsl_LIBRARIES} 
  ${zlib_LIBRARIES}
  CGAL CGAL_ImageIO dolfin metis
  gmp boost_system )
