
/**
 * A simple timer for calculating user time.
 */

#ifndef _UNIX_TIMER_H_
#define _UNIX_TIMER_H_

#ifdef __cplusplus
extern "C" { 
#endif

#include <time.h>
#include <time.h>

// For execution timing
#if !defined(SERIAL) && defined(CILKVIEW_TIMING)
	#include <cilktools/cilkview.h>
#endif

// MSVC defines this in winsock2.h!?
typedef struct timeval {
	long tv_sec;
	long tv_usec;
} timeval;

typedef struct timeval timer_time;
typedef struct {
	timer_time start_time;
} timer_id;

#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <stdint.h> // portable: uint64_t   MSVC: __int64



int gettimeofday(struct timeval * tp, struct timezone * tzp)
{
	// Note: some broken versions only have 8 trailing zero's, the correct epoch has 9 trailing zero's
	// This magic number is the number of 100 nanosecond intervals since January 1, 1601 (UTC)
	// until 00:00:00 January 1, 1970
	static const uint64_t EPOCH = ((uint64_t) 116444736000000000ULL);

	SYSTEMTIME  system_time;
	FILETIME    file_time;
	uint64_t    time;

	GetSystemTime( &system_time );
	SystemTimeToFileTime( &system_time, &file_time );
	time =  ((uint64_t)file_time.dwLowDateTime )      ;
	time += ((uint64_t)file_time.dwHighDateTime) << 32;

	tp->tv_sec  = (long) ((time - EPOCH) / 10000000L);
	tp->tv_usec = (long) (system_time.wMilliseconds * 1000);
	return 0;
}

static inline void _startTimer(unsigned long long *start){
    struct timeval t;
    gettimeofday(&t, 0);
    *start = (unsigned long long) ((t.tv_sec)*1000000ll) + t.tv_usec;
}

static inline void _stopTimer(unsigned long long *start, float *elapsed){
    struct timeval now;
    gettimeofday(&now, 0);
	unsigned long long nowll = (unsigned long long) ((now.tv_sec)*1000000ll) + now.tv_usec;
	// nowll -= *start;
	*elapsed = (float) ((nowll - *start) / 1000000.0);
	*start = nowll;
}

static inline void _stopTimerAddElapsed(unsigned long long *start, float *elapsed){
    struct timeval now;
    gettimeofday(&now, 0);
	unsigned long long nowll = (unsigned long long) ((now.tv_sec)*1000000ll) + now.tv_usec;
	nowll -= *start;
	*elapsed += (float) (nowll / 1000000.0);
}








#ifdef __cplusplus
} 
#endif

#endif

