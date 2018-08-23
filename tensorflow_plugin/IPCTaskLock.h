#ifndef __IPCTASKLOCK_H_
#define __IPCTASKLOCK_H_

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>
#include <sched.h>
#include <sys/mman.h>
#include <atomic>
#include <iostream>

// state -> 0 master is running
// state -> 1 master is ceding, waiting for worker
// state -> 2 worker is running, master is ceding
// state -> 3 worker is ceding, waiting for master
// state -> 4 call exit
// state -> 5 exit acknowledged
struct IPCTaskLock {
  std::atomic_char* _latch;
  IPCTaskLock() {
    _latch = (std::atomic_char*)mmap(nullptr, sizeof(std::atomic_char),
                                     PROT_READ | PROT_WRITE,
                                     MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    _latch->store(0);
  }

  IPCTaskLock(IPCTaskLock&& other) { *this = std::move(other); }

  IPCTaskLock& operator=(IPCTaskLock&& other) {
    if (_latch) munmap(_latch, sizeof(std::atomic_char));
    _latch = other._latch;
    other._latch = nullptr;
    return *this;
  }
  ~IPCTaskLock() {
    if (_latch) munmap(_latch, sizeof(std::atomic_char));
  }

  // Wait for worker process to complete
  void await() {
    // no competition here
    _store(1);
    // now latch is state 1

    // move from state 3 to 0 (worker must go from 1 to 2)
    _change_state(3, 0);
  }

  void exit() {
    //_change_state(0, 4);
  }

  // start work
  bool start() {
    // move from state 1 to 2
    return _change_state(1, 2);
  }

  void end() {
    // move from state 2 to 3
    // no competition
    _store(3);
  }

 private:
  void _store(char state) { _latch->store(state, std::memory_order_release); }

  /*
   * https://stackoverflow.com/a/26583492/2392535
   *
   *  vars: expected, desired, latch
   *  the while(!latch->compare_exchange_weak(expected, desired..)...
   *  is equivalent to this:
   * if(latch == expected)
   *   latch = desired
   * else
   *   continue loop
   *
   */
  bool _change_state(char state_start, char state_end) {
    char expected = state_start;
    while (!_latch->compare_exchange_weak(expected, state_end,
                                          std::memory_order_acquire)) {
      // if(expected == 4)
      // return false;
      expected = state_start;
    }
    return true;
  }
};

void export_IPCTaskLock(pybind11::module& m);

#endif  //__IPCTASKLOCK_H_
