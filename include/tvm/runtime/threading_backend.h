/*!
 *  Copyright (c) 2018 by Contributors
 * \file tvm/runtime/threading_backend.h
 * \brief Utilities for manipulating thread pool threads.
 */
#ifndef TVM_RUNTIME_THREADING_BACKEND_H_
#define TVM_RUNTIME_THREADING_BACKEND_H_

#include <functional>
#include <memory>
#include <vector>

namespace tvm {
namespace runtime {
namespace threading {

/*!
 * \brief A platform-agnostic abstraction for managing a collection of
 *        thread pool threads.
 */
class ThreadGroup {
 public:
  class Impl;

   /*!
    * \brief Creates a collection of threads which run a provided function.
    *
    * \param num_workers The total number of worker threads in this group.
             Includes main thread if `exclude_worker0 = true`
    * \param worker_callback A callback which is run in its own thread.
             Receives the worker_id as an argument.
    * \param exclude_worker0 Whether to use the main thread as a worker.
    *        If  `true`, worker0 will not be launched in a new thread and
    *        `worker_callback` will only be called for values >= 1. This
    *        allows use of the main thread as a worker.
    */
  ThreadGroup(int num_workers,
              std::function<void(int)> worker_callback,
              bool exclude_worker0 = false);
  ~ThreadGroup();

   /*!
    * \brief Blocks until all non-main threads in the pool finish.
    */
  void Join();

<<<<<<< HEAD
=======
  enum AffinityMode : int {
    kBig = 1,
    kLittle = -1,
  };

  /*!
   * \brief configure the CPU id affinity
   *
   * \param mode The preferred CPU type (1 = big, -1 = little).
   * \param nthreads The number of threads to use (0 = use all).
   * \param exclude_worker0 Whether to use the main thread as a worker.
   *        If  `true`, worker0 will not be launched in a new thread and
   *        `worker_callback` will only be called for values >= 1. This
   *        allows use of the main thread as a worker.
   *
   * \return The number of workers to use.
   */
  int Configure(AffinityMode mode, int nthreads, bool exclude_worker0);

>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199
 private:
  Impl* impl_;
};

/*!
 * \brief Platform-agnostic no-op.
 */
void Yield();

/*!
 * \return the maximum number of effective workers for this system.
 */
int MaxConcurrency();

<<<<<<< HEAD
=======

>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199
}  // namespace threading
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_THREADING_BACKEND_H_
