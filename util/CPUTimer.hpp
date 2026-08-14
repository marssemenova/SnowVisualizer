
/**
 * A simple timer for calculating user time.
 */

#ifndef _CPU_TIMER_H_
#define _CPU_TIMER_H_

#include <chrono>

#define PROFILING true // used for timing
#define NUM_FRAMES 100

typedef std::chrono::time_point<std::chrono::high_resolution_clock> time_point;


static int frameCount = 0;

static inline time_point startTimer() {
	return std::chrono::high_resolution_clock::now();
}

static inline time_point stopTimer() {
	return std::chrono::high_resolution_clock::now();
}

/**
 * Return elapsed time in ms
 * @param start
 * @param end
 * @return time in ms
 */
static inline float elapsedTime(const time_point &start, const time_point &end) {
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	return static_cast<float>(duration.count() / 1000.0);
}


#endif

