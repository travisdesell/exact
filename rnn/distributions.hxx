#ifndef RNN_DISTRIBUTIONS
#define RNN_DISTRIBUTIONS

#include "rnn_genome.hxx"
#include "rnn_recurrent_edge.hxx"


class RecDepthFrequencyTable {
    public:
        int32_t[] frequencies;
        
        RecDepthFrequencyTable(vector<RNN_Genome*> &genomes, 
                                int32_t _min_recurrent_depth, int32_t max_recurrent_depth);

        RecDepthFrequencyTable(vector<vector<RNN_Genome*>> &islands, 
                                int32_t _min_recurrent_depth, int32_t max_recurrent_depth);

        ~RecDepthFrequencyTable();

        int32_t &operator[] (int32_t i);

    private:
        void count_island_frequencies(vector<RNN_Genome*> &genomes);
};

class Distribution {
    public:
        virtual int32_t sample() = 0;
};

class RecDepthNormalDist : public Distribution {
    private:
        double mean;
        double sd;
        int32_t min, max;
    public:
        NormalDist(vector<RNN_Genome*> &genomes, 
                    int32_t min_recurrent_depth, int32_t max_recurrent_depth); 

        NormalDist(vector<vector<RNN_Genome*>> &islands,
                        int32_t min_recurrent_depth, int32_t max_recurrent_depth);

    private:
        void calculate_parameters(RedDepthFrequencyTable freqs&);
};

#endif
