#ifndef COMMON_H
#define COMMON_H

#include <time.h>
#include <sys/time.h>

#define CHECKERROR() \
  { \
    hipError_t __err = hipGetLastError(); \
    if (__err != hipSuccess) \
    { \
      fprintf(stderr, "Fatal error: %s at %s:%d\n", hipGetErrorString(__err), __FILE__, __LINE__); \
      fprintf(stderr, "*** FAILED - ABORTING\n"); \
      exit(1); \
    } \
  }

#define CHECK(cmd) \
  { \
    (cmd); \
    CHECKERROR(); \
  }

typedef long long   int64;

namespace Common
{
    struct Timer
    {
        Timer() {}

        void start();

        void stop();

        float seconds() const;

    private:
        int64 m_start;
        int64 m_stop;
        int64 m_start_ns;
        int64 m_stop_ns;
    };
    void Timer::start()
    {
        timeval _time;
        gettimeofday(&_time,NULL);
        m_start    = _time.tv_sec;
        m_start_ns = _time.tv_usec;
    }
    void Timer::stop()
    {
        timeval _time;
        gettimeofday(&_time,NULL);
        m_stop    = _time.tv_sec;
        m_stop_ns = _time.tv_usec;
    }

    float Timer::seconds() const
    {
        if (m_stop_ns < m_start_ns)
            return float( double(m_stop - m_start - 1) + double(1000000 + m_stop_ns - m_start_ns)*1.0e-6 );
        else
            return float( double(m_stop - m_start) + double(m_stop_ns - m_start_ns)*1.0e-6 );
    }
}
#endif