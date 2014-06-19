CXXFLAGS +=  -DTRACE=100 -fPIC -pedantic -Wall
LDFLAGS   =  -shared
#
# Libraries and includes
#
INC     = -I$(FIJEE)/ 
LIB_DIR = 
LIBS    = -lgsl #-pthread 
#
# Sources
#
SRC        = Brain_rhythm.cxx Jansen_Rit_1995.cxx 
OBJ        = $(SRC:.cxx=.o)
#
#
#
EXEC = libfijee_biophysics.so
#
#
#
all: $(EXEC) 

$(EXEC): $(OBJ)
	$(CXX) $(LDFLAGS) $(INC) -o $@  $?


%.o: %.cxx
	$(CXX) $(CXXFLAGS) $(INC) -o $@ -c $<



clean:
	rm $(OBJ) $(EXEC)
