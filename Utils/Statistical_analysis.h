#ifndef statistical_analysis_H_
#define statistical_analysis_H_
//http://franckh.developpez.com/tutoriels/outils/doxygen/
/*!
 * \file Statistical_analysis.h
 * \brief brief describe 
 * \author Yann Cobigo
 * \version 0.1
 */
#include <iostream>
#include <fstream>      // std::ifstream, std::ofstream
#include <sstream>
/*! \namespace Utils
 * 
 * Name space for our new package
 *
 */
namespace Utils
{
  /*! \class Statistical_analysis
   * \brief classe representing whatever
   *
   *  This class is an example of class I will have to use
   */
  class Statistical_analysis
  {
  private:
    std::ofstream file_;


  protected:
    //! Stream populating the output file.
    std::stringstream output_stream_;

    
  public:
    /*!
     *  \brief Default Constructor
     *
     *  Constructor of the class Statistical_analysis
     *
     */
    Statistical_analysis(){};
    /*!
     *  \brief Copy Constructor
     *
     *  Constructor is a copy constructor
     *
     */
    Statistical_analysis( const Statistical_analysis& ){};
    /*!
     *  \brief Destructeur
     *
     *  Destructor of the class Statistical_analysis
     */
    virtual ~Statistical_analysis(){/* Do nothing */};
    /*!
     *  \brief Operator =
     *
     *  Operator = of the class Statistical_analysis
     *
     */
    Statistical_analysis& operator = ( const Statistical_analysis& ){return *this;};
    
  public:
    /*!
     */
    void Make_output_file( const char* file_name)
    {
      file_.open( file_name );
      file_ << output_stream_.rdbuf();
      file_.close();  
    };

  private:
    /*!
     */
    virtual void Make_analysis() = 0;
 };
  /*!
   *  \brief Dump values for Statistical_analysis
   *
   *  This method overload "<<" operator for a customuzed output.
   *
   *  \param Point : new position to add in the list
   */
  std::ostream& operator << ( std::ostream&, const Statistical_analysis& );
};
#endif
