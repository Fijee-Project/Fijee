#include <iostream>
#include <fijee.h>

// 
// g++ -std=c++11 -DTRACE=100 -I. -I/home/cobigo/devel/C++/UCSF/Dev016/  Brain_rhythm.cxx Jansen_Rit_1995.cxx Molaee_Ardekani_Wendling_2009.cxx Wendling_2002.cxx Population.cxx Leadfield_matrix.cxx EEG_simulation.cxx  main.cxx -L/home/cobigo/devel/C++/UCSF/Dev016/Utils/pugi/ -lpugixml -L/home/cobigo/devel/C++/UCSF/Dev016/Utils/Compression -lfijee_compression -lgsl -lz

//

// Biophysics::Jansen_Rit_1995
// Biophysics::Wendling_2002
// Biophysics::Molaee_Ardekani_Wendling_2009
int main()
{
  Biophysics::Brain_rhythm_models<Biophysics::Wendling_2002, 4> test_brain_rhythm;
  // 
  test_brain_rhythm.modelization("/home/cobigo/subjects/GazzDCS0004mgh_GPU4/fem/output/");
  test_brain_rhythm.modelization_at_electrodes("/home/cobigo/subjects/GazzDCS0004mgh_GPU4/fem/output/"/*, 0.000005*/);
  test_brain_rhythm.output();


  return 1;
}
