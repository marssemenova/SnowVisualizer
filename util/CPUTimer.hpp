
/**
 * CPUTimer.hpp - A simple timer for calculating user time.
 *
 * @author Dr. Alexander Brandt
 */

#ifndef _CPU_TIMER_H_
#define _CPU_TIMER_H_

#include <chrono>

typedef std::chrono::time_point<std::chrono::high_resolution_clock> time_point;

static time_point startTimer() {
	return std::chrono::high_resolution_clock::now();
}

static time_point stopTimer() {
	return std::chrono::high_resolution_clock::now();
}

/**
 * Return elapsed time in ms.
 * @param start
 * @param end
 * @return Time in ms.
 */
static float elapsedTime(const time_point &start, const time_point &end) {
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	return static_cast<float>(duration.count() / 1000.0);
}

#endif

