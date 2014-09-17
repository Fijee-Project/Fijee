FIJEE=/home/cobigo/devel/C++/UCSF/Dev016/
CXXFLAGS  =  -std=c++11
#
CXXFLAGS +=  -DTRACE=100 -fPIC -pedantic -Wall
LDFLAGS   =  -shared
#
# Libraries and includes
#
INC     = -I$(FIJEE)/ 
LIB_DIR = 
LIBS    = 
#
# Sources
#
SRC        = Electrodes_tACS.cxx
OBJ        = $(SRC:.cxx=.o)
#
#
#
EXEC = libfijee_electrodes.so
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
