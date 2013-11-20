#ifndef _PHYSICAL_MODEL_H
#define _PHYSICAL_MODEL_H
//
//
//
//using namespace dolfin;
//
/*!
 * \file Physical_model.h
 * \brief brief describe 
 * \author Yann Cobigo
 * \version 0.1
 */

/*! \namespace Solver
 * 
 * Name space for our new package
 *
 */
namespace Solver
{
  /*! \class Physical_model
   * \brief classe representing the dipoles distribution
   *
   *  This class is an example of class I will have to use
   */
  class Physical_model
  {
  public:
   /*!
     */
    virtual void solver_loop() = 0;
  };
}
#endif
