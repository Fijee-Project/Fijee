# - FindFijee.cmake
#
# Author: Yann Cobigo
#
#
# The following variables will be exported:
# 
# Fijee_INCLUDE_DIR        - the directory that contains nifti1.h
# Fijee_UTILS_LIBRARY      - the libFijeeUtils.so library
# Fijee_BIOPHYSICS_LIBRARY - the libFijeeBiophysics.so library
# Fijee_FOUND              - TRUE if and only if ALL other variables have 
#                            correct values.
#

#
# INCLUDE directory
FIND_PATH( Fijee_INCLUDE_DIR
  NAMES fijee.h
  PATH_SUFFIXES Fijee
  DOC "The include directory containing fijee.h" )
#
# LIBRARY files
FIND_LIBRARY( Fijee_UTILS_LIBRARY
  NAMES FijeeUtils
  DOC "The library file libFijeeUtils.so" )
#
# LIBRARY files
FIND_LIBRARY( Fijee_BIOPHYSICS_LIBRARY
  NAMES FijeeBiophysics
  DOC "The library file libFijeeBiophysics.so" )

#
# handle the QUIETLY and REQUIRED arguments and set PNG_FOUND to TRUE if 
# all listed variables are TRUE
INCLUDE( FindPackageHandleStandardArgs )
#
FIND_PACKAGE_HANDLE_STANDARD_ARGS(Fijee
  "Cannot find package Fijee. Did you install the package?"
  Fijee_INCLUDE_DIR
  Fijee_UTILS_LIBRARY
  Fijee_BIOPHYSICS_LIBRARY )

# these variables are only visible in 'advanced mode' 
MARK_AS_ADVANCED( Fijee_INCLUDE_DIR
  Fijee_UTILS_LIBRARY
  Fijee_BIOPHYSICS_LIBRARY )
