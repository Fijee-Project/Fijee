CXXFLAGS +=  -fPIC -pedantic -Wall
LDFLAGS   = -shared
#
# Libraries and includes
#
INC     = 
LIB_DIR = 
LIBS    = 
#
# Sources
#
SRC        = pugixml.cpp
OBJ        = $(SRC:.cpp=.o)
#
#
#
EXEC = libpugixml.so
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
