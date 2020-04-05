#ifndef EXAMM_THOMPSON_SAMPLING_HXX
#define EXAMM_THOMPSON_SAMPLING_HXX

#include <random>
using std::minstd_rand0;

class ThompsonSampling {
    protected:
        int32_t n_actions;
    public:
        /**
         * Creates a new ThompsonSampling object with _n_actions actions (identified by integer keys 0 through _n_actions - 1)
         **/
        ThompsonSampling(int32_t _n_actions) : n_actions(_n_actions) {}
        virtual ~ThompsonSampling() {}

        virtual int32_t sample_action(minstd_rand0 &generator) = 0;
        virtual void update(int32_t action, double reward) = 0;
        virtual void print(vector<int> &possible_node_types, const string node_types[]) = 0;
};

#endif
