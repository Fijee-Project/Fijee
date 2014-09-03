CXXFLAGS +=  -fPIC -pedantic -Wall
LDFLAGS   =  -shared
#
# Libraries and includes
#
INC     = 
LIB_DIR = 
LIBS    = -lz
#
# Sources
#
SRC        = Fijee_compression.cxx
OBJ        = $(SRC:.cxx=.o)
#
#
#
EXEC = libfijee_compression.so
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
