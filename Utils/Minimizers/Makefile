CXXFLAGS +=  -fPIC -pedantic -Wall
LDFLAGS   =  -shared
#
# Libraries and includes
#
INC     = -I$(FIJEE)/ -I$(EIGEN3)/include/eigen3/ 
LIB_DIR = 
LIBS    = 
#
# Sources
#
SRC        = Downhill_simplex.cxx
OBJ        = $(SRC:.cxx=.o)
#
#
#
EXEC = libfijee_minimizers.so
#
#
#
all: $(EXEC) 

$(EXEC): $(OBJ)
	$(CXX) $(LDFLAGS) $(INC) -o $@  $<


%.o: %.cxx
	$(CXX) $(CXXFLAGS) $(INC) -o $@ -c $<



clean:
	rm $(OBJ) $(EXEC)
