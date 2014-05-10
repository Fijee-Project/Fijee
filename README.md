Fijee
=====

Forward inverse / Electroencephalogram software. Building a new forward solution.
To get started:
1. create a SImplex instance
	"Simplex *sx;"
2. construct the instance with a function R^n->R. In the example code, the function double banana(Vector3d& point)
   returns a double
	"sx = new Simplex(banana);"
3. run the algorithm and store the result in a Vector3d vc
	"Vector3d vc = sx->findOptimal(200);"