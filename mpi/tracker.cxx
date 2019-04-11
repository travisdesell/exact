#include <cmath>

#include <vector>
using std::vector;

#include "stdint.h"
#include "tracker.hxx"

Tracker::Tracker() {
    count = 0;
    sum = 0.0;
    _min = 999999;
    _max = -999999;
}

void Tracker::track(double value) {
    if (value < _min) _min = value;
    if (value > _max) _max = value;

    count++;
    sum += value;

    values.push_back(value);
}

double Tracker::min() const {
    return _min;
}

double Tracker::max() const {
    return _max;
}

double Tracker::avg() const {
    return sum / count;
}

double Tracker::stddev() {
    double _avg = avg();
    double _stddev = 0;

    for (int i = 0; i < values.size(); i++) {
        double tmp = (values[i] - _avg);
        _stddev += tmp * tmp;
    }

    _stddev = sqrt(_stddev / (values.size() - 1));

    return _stddev;
}

double Tracker::correlate(Tracker &other) {
    double avg1 = avg();
    double avg2 = other.avg();

    double stddev1 = stddev();
    double stddev2 = other.stddev();

    double correlation = 0.0;

    for (int i = 0; i < values.size(); i++) {
        correlation += (values[i] - avg1) * (other.values[i] - avg2);
    }

    correlation /= (count - 1) * stddev1 * stddev2;

    return correlation;
}

