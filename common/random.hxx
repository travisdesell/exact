#ifndef EXACT_RANDOM_HXX
#define EXACT_RANDOM_HXX

#include <iostream>
using std::ostream;
using std::istream;

#include <random>
using std::minstd_rand0;

#include <vector>
using std::vector;

void fisher_yates_shuffle(minstd_rand0 &generator, vector<int> &v);
void fisher_yates_shuffle(minstd_rand0 &generator, vector<long> &v);

float random_0_1(minstd_rand0 &generator);

class NormalDistribution {
    private:
        bool generate;
        float z0;
        float z1;

    public:

        NormalDistribution();

        float random(minstd_rand0 &generator, float mu, float sigma);

        friend ostream &operator<<(ostream &os, const NormalDistribution &normal_distribution);
        friend istream &operator>>(istream &is, NormalDistribution &normal_distribution);

        bool operator==(const NormalDistribution &other) const;
        bool operator!=(const NormalDistribution &other) const;
};

#endif
