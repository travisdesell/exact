#ifndef EXAMM_BETA_THOMPSON_SAMPLING_HXX
#define EXAMM_BETA_THOMPSON_SAMPLING_HXX

#include <vector>
using std::vector;

#include <string>
using std::string;

#include "thompson_sampling.hxx"

using std::gamma_distribution;

class BetaThompsonSampling : public ThompsonSampling {
    private:
        vector<double> alphas;
        vector<double> betas;
        double decay_rate;

    public:
        /**
         * Creates a new ThompsonSampling object with _n_actions actions (identified by integer keys 0 through _n_actions - 1)
         **/
        BetaThompsonSampling(int32_t _n_actions, double _decay_rate);
        ~BetaThompsonSampling();

        int32_t sample_action(minstd_rand0 &generator);
        void update(int32_t action, double reward);
        void print(vector<int> &possible_node_types, const string node_types[]);
};

#endif
