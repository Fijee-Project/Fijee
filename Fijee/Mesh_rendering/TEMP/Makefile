#
# make
#
# ./test cpu  -> only cpu
# ./test gpu  -> only gpu
# ./test both -> only cpu and gpu
#
#
CXX      = g++ -m64 -O3 
CXXFLAGS = -std=c++0x 
INC      = -I../
#########
# CUDA   #
LIBS += -L/usr/local/cuda-5.0/lib64  -lcudart

######
SRC = test.cxx
OBJ = $(SRC:.cxx=.o)

EXEC = test

all: $(EXEC)

$(EXEC): $(OBJ)
	$(CXX) $(OBJ) ../CUDA_Conductivity_matching.o $(LIBS) -o $@


%.o: %.cxx
	$(CXX) $(CXXFLAGS) $(INC) -o $@ -c $<



clean: 
	rm *~ test $(OBJ) 
