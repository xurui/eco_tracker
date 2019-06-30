#ifndef _TIME_LOG_HPP_
#define _TIME_LOG_HPP_

#include <string>
#include <vector>
#include <map>
#include <math.h>
#include <queue>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <sys/time.h>

class timer
{
public:
    timer() {
		last_time.tv_sec = 0;
		last_time.tv_usec = 0;
		curr_time = last_time;
	};

    ~timer() {};

    void reset() { gettimeofday(&last_time, NULL); };

    double ms_delay() {
		gettimeofday(&curr_time, NULL);
		double ms_used = 1000.0*(curr_time.tv_sec - last_time.tv_sec) + (curr_time.tv_usec - last_time.tv_usec)/1000.0;
		last_time = curr_time;
		return ms_used;
	};

private:
    struct timeval last_time;
    struct timeval curr_time;
};

#endif