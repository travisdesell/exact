#ifndef EXAMM_TRACKER_HXX
#define EXAMM_TRACKER_HXX

#include <vector>
using std::vector;

#include "stdint.h"

class Tracker {
    private:
        int32_t count;
        double _min;
        double sum;
        double _max;

        vector<double> values;

    public:
        Tracker();

        void track(double value);

        double min() const;
        double max() const;
        double avg() const;
        double stddev();

        double correlate(Tracker &other);
};

#endif
