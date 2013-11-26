# cible: dependance
# ( tab ) commandes
#
# genericity
# %.o: %.cxx
# ( tab ) commandes
#	
#
############
# ENV
# C++
CXX = g++
CXXFLAGS = 
# C
CC = gcc
CFLAGS = -W -Wall -ansi -pedantic
# Executable
EXEC = Point_vector
# Sources and objects
SRC =  main.cxx 
OBJ = $(SRC:.cxx=.o)
# Includes
INCLUDES = -I../../ -I/home/cobigo/devel/CPP/CGAL/install/include/
#Libraries
LIB += -L/home/cobigo/devel/CPP/CGAL/install/lib -lCGAL_Core -lCGAL
# Link options
LDFLAGS =


# $@ 	Le nom de la cible
# $< 	Le nom de la première dépendance
# $^ 	La liste des dépendances
# $? 	La liste des dépendances plus récentes que la cible
# $* 	Le nom du fichier sans suffixe



all: $(EXEC)

Point_vector: $(OBJ)
	$(CXX)  $(INCLUDES) $(LIB) -o $@ $^ ../../Point_vector.o ../../Point.o 

%.o: %.cxx
	$(CXX) -frounding-math $(INCLUDES) -o $@ -c $< $(CXXFLAGS)


clean:
	rm -rf *.o $(EXEC)
