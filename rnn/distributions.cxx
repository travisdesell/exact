#include "distributions.hxx"

RecDepthFrequencyTable::RecDepthFrequencyTable(
        vector<RNN_Genome*> &genomes, int32_t _min_recurrent_depth, int32_t max_recurrent_depth) {
    frequencies = new int32_t[max_recurrent_depth + 1];
    count_island_frequencies(genomes);
}

RecDepthFrequencyTable::RecDepthFrequencyTable(
        vector<vector<RNN_Genome*>> &islands, int32_t _min_recurrent_depth, int32_t max_recurrent_depth) {
    frequencies = new int32_t[max_recurrent_depth + 1];
    for (int32_t i = 0; i < islands.size(); i += 1) count_island_frequencies(islands[i]);
}

RedDepthFrequencyTable::~RecDepthFrequencyTable() {
    delete frequencies;
}

int32_t & RedDepthFrequencyTable::operator[] (int32_t i) {
    return frequencies[i];
}

void RedDepthFrequencyTable::count_island_frequencies(vector<RNN_Genome*> &genomes) {
    for (int32_t gi = 0; gi < genomes.size(); gi += 1) {
        RNN_Genome *genome = genomes[gi];
        for (int32_t ei = 0; ei < genome->recurrent_edges.size(); ei += 1) {
            RNN_Recurrent_Edge *edge = genome->edges[ei];
            // TODO: Figure out if i should count disabled edges as well?
            // intuition says no.
            if (edge->enabled) {
                frequencies[edge->recurrent_depth] += 1;
            }
        }
    }
}

RecDepthNormalDist::RecDepthNormalDist(vector<RNN_Genome*> &genomes, 
            int32_t min_recurrent_depth, int32_t max_recurrent_depth) {
    min = min_recurrent_depth;
    max = max_recurrent_depth;

    int32_t sum = 0;
    int32_t n_samples = 0;
    RecDepthFrequencyTable freqs(genomes, min_recurrent_depth, max_recurrent_depth);

    calculate_parameters(freqs);
}

RecDepthNormalDist::RecDepthNormalDist(vector<vector<RNN_Genome*>> &islands,
                int32_t min_recurrent_depth, int32_t max_recurrent_depth) {
    min = min_recurrent_depth;
    max = max_recurrent_depth;

    int32_t sum = 0;
    int32_t n_samples = 0;
    RecDepthFrequencyTable freqs(islands, min_recurrent_depth, max_recurrent_depth);

    calculate_parameters(freqs);
}

void RecDepthNormalDist::calculate_parameters(RedDepthFrequencyTable freqs&) {
    // Yes, inclusive range
    for (int32_t i = min; i <= max; i += 1) {
        assert(sum >= 0); // Check for overflows! 
        int32_t f = freqs[i];
        n_samples += f;
        sum += (f * i);
    }
    mean = ((double) sum) / ((double) n_samples);
    
    double total_sq_deviation = 0.0;
    for (int32_t i = min; i <= max; i += 1) {
        double f = freqs[i];
        double deviation = ((double) i) - mean;
        total_sq_deviation += f * deviation * deviation;
    }
    sd = sqrt(total_sq_deviation / n_samples);
}
