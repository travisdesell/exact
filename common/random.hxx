#ifndef EXACT_RANDOM_HXX
#define EXACT_RANDOM_HXX

#include <random>
using std::minstd_rand0;

#include <vector>
using std::vector;

void fisher_yates_shuffle(minstd_rand0 &generator, vector<long> &v);

double random_0_1(minstd_rand0 &generator);

class NormalDistribution {
    private:
        bool generate;
        double z0;
        double z1;

    public:

        NormalDistribution();

        double random(minstd_rand0 &generator, double mu, double sigma);

        friend ostream &operator<<(ostream &os, const NormalDistribution &normal_distribution);
        friend istream &operator>>(istream &is, NormalDistribution &normal_distribution);
};

#endif
