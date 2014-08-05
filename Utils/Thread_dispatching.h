//  Copyright (c) 2014, Yann Cobigo 
//  All rights reserved.     
//   
//  Redistribution and use in source and binary forms, with or without       
//  modification, are permitted provided that the following conditions are met:   
//   
//  1. Redistributions of source code must retain the above copyright notice, this   
//     list of conditions and the following disclaimer.    
//  2. Redistributions in binary form must reproduce the above copyright notice,   
//     this list of conditions and the following disclaimer in the documentation   
//     and/or other materials provided with the distribution.   
//   
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND   
//  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED   
//  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE   
//  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR   
//  ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES   
//  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;   
//  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND   
//  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT   
//  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS   
//  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.   
//     
//  The views and conclusions contained in the software and documentation are those   
//  of the authors and should not be interpreted as representing official policies,    
//  either expressed or implied, of the FreeBSD Project.  
#ifndef THREAD_DISPATCHING_H
#define THREAD_DISPATCHING_H
#include <vector>
#include <queue>
#include <memory>
#include <thread>
#include <chrono>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <stdexcept>
/*!
 * \file Thread_dispatching.h
 * \brief brief describe 
 * \author Yann Cobigo
 * \version 0.1
 */

/*! \namespace Utils
 * 
 * Name space for our new package
 *
 */
namespace Utils
{
  /*! \class Thread_dispatching
   * \brief classe representing whatever
   *
   *  This class is an example of class I will have to use
   */
  class Thread_dispatching {
  public:
    /*!
     *  \brief Default Constructor
     *
     *  Constructor of the class Thread_dispatching
     *
     */
    Thread_dispatching(size_t);
    /*!
     */
    template<class F, class... Args>
      auto enqueue(F&& f, Args&&... args)
      -> std::future<typename std::result_of<F(Args...)>::type>;
    /*!
     *  \brief Destructor
     *
     *  Destructor of the class Thread_dispatching
     *
    */
    ~Thread_dispatching();

  private:
    //! Vector keeping track of threads for the join process 
    std::vector< std::thread > workers;
    //! Queue for queueing the tasks dispatched in each threads
    std::queue< std::function<void()> > tasks;
    
    // synchronization
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;
  };
  // 
  // the constructor just launches some amount of workers
  //
  inline Thread_dispatching::Thread_dispatching(size_t threads)
    : stop(false)
  {
    //
    for( size_t i = 0 ; i < threads ; ++i )
      workers.emplace_back(
			   [this]
			   {
			     while( true )
			       {
				 //
				 std::unique_lock< std::mutex > lock( this->queue_mutex );
				 //
				 while( !this->stop && this->tasks.empty() )
				   this->condition.wait(lock);
				 //
				 if( this->stop && this->tasks.empty() )
				   return;
				 //
				 std::function< void() > task(this->tasks.front());
				 this->tasks.pop();
				 lock.unlock();
				 task();
			       }
			   }
			   );
  }
  //
  // add new work item to the pool
  // 
  template<class F, class... Args>
    auto Thread_dispatching::enqueue( F&& f, Args&&... args )
    -> std::future<typename std::result_of<F(Args...)>::type>
  {
    typedef typename std::result_of< F(Args...) >::type return_type;
    
    // don't allow enqueueing after stopping the pool
    if( stop )
      throw std::runtime_error("enqueue on stopped Thread_dispatching");

    auto task = std::make_shared< std::packaged_task< return_type() > >(
								      std::bind(std::forward< F >(f), std::forward< Args >(args)...)
								      );
    
    std::future< return_type > res = task->get_future();
    {
      std::unique_lock< std::mutex > lock(queue_mutex);
      tasks.push([task](){ (*task)(); });
    }
    condition.notify_one();
    return res;
  }
  //
  // the destructor joins all threads
  // 
  inline Thread_dispatching::~Thread_dispatching()
  {
    {
      std::unique_lock< std::mutex > lock( queue_mutex );
      stop = true;
    }
    condition.notify_all();
    for( size_t i = 0 ; i < workers.size() ; ++i )
      {
	workers[i].join();
	// pospone the next launch 0.05 second
	std::this_thread::sleep_for( std::chrono::microseconds(50) );

      }
  }
}
#endif
