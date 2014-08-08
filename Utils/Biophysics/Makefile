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
SRC        = Population.cxx Leadfield_matrix.cxx Brain_rhythm.cxx Jansen_Rit_1995.cxx Molaee_Ardekani_Wendling_2009.cxx Wendling_2002.cxx EEG_simulation.cxx 
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
