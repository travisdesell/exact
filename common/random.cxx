#include <cstdlib>
#include <cmath>
#include <limits>
#include <cstdint>

#include <iomanip>
using std::setprecision;

#include <iostream>
using std::ostream;
using std::istream;

#include <random>
using std::minstd_rand0;

#include <vector>
using std::vector;

#include "random.hxx"

double random_0_1(minstd_rand0 &generator) {
    return ((double)generator() - (double)generator.min()) / ((double)generator.max() - (double)generator.min());
}

void fisher_yates_shuffle(minstd_rand0 &generator, vector<long> &v) {
    for (int32_t i = v.size() - 1; i > 0; i--) {
        double t = ((double)generator() - (double)generator.min()) / ((double)generator.max() - (double)generator.min());
        t *= (double)i - 1.0;

        int32_t target = (int32_t)t;

        //cerr << "target: " << target << endl;

        long temp = v[target];
        v[target] = v[i];
        v[i] = temp;
    }
}

NormalDistribution::NormalDistribution() {
    generate = true;
    z0 = 0;
    z1 = 0;
}

double NormalDistribution::random(minstd_rand0 &generator, double mu, double sigma) {
    const double epsilon = std::numeric_limits<double>::min();
    const double two_pi = 2.0*3.14159265358979323846;

    generate = !generate;

    if (!generate) {
        return z1 * sigma + mu;
    }

    double u1, u2;
    do {
        u1 = ((double)generator() - (double)generator.min()) / ((double)generator.max() - (double)generator.min());
        u2 = ((double)generator() - (double)generator.min()) / ((double)generator.max() - (double)generator.min());
    } while ( u1 <= epsilon );

    z0 = sqrt(-2.0 * log(u1)) * cos(two_pi * u2);
    z1 = sqrt(-2.0 * log(u1)) * sin(two_pi * u2);
    return z0 * sigma + mu;
}

ostream &operator<<(ostream &os, const NormalDistribution &normal_distribution) {
    os << normal_distribution.generate << " " << setprecision(15) << normal_distribution.z0 << " " << setprecision(15) << normal_distribution.z1;

    return os;
}

istream &operator>>(istream &is, NormalDistribution &normal_distribution) {
    is >> normal_distribution.generate >> normal_distribution.z0 >> normal_distribution.z1;

    return is;
}


