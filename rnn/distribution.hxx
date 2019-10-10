#ifndef RNN_DISTRIBUTION
#define RNN_DISTRIBUTION

class Distribution {
    public:
        Distribution() {
            // Needed to sample distribution
            unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
            rng = mt19937(seed);
        }
        virtual ~Distribution() { };
        
        virtual int32_t sample() = 0;
    protected:
        // Replaced with higher quality rng mt19937 because we want the samples to be random,
        // unsurprisingly.
        // minstd_rand0 rng;
        mt19937 rng;
};

#endif
