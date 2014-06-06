set(PUGI_SOURCES pugixml.cpp)


set(PUGI_GCC_COMPILER_FLAGS "-fPIC -pedantic -Wall")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${PUGI_GCC_COMPILER_FLAGS}")

add_library(pugi SHARED ${PUGI_SOURCES})

