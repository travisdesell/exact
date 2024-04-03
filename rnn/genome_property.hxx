#ifndef GENOME_PROPERTY_HXX
#define GENOME_PROPERTY_HXX

#include <vector>
using std::vector;

#include <string>
using std::string;

#include "rnn/rnn_genome.hxx"
#include "time_series/time_series.hxx"

class GenomeProperty {
   private:
    int32_t bp_iterations;
    bool use_dropout;
    double dropout_probability;
    int32_t min_recurrent_depth;
    int32_t max_recurrent_depth;
    bool classification;
    
    // TimeSeriesSets *time_series_sets;
    int32_t number_inputs;
    int32_t number_outputs;
    vector<string> input_parameter_names;
    vector<string> output_parameter_names;

    string normalize_type;
    map<string, double> normalize_mins;
    map<string, double> normalize_maxs;
    map<string, double> normalize_avgs;
    map<string, double> normalize_std_devs;

   public:
    GenomeProperty();
    void generate_genome_property_from_arguments(const vector<string>& arguments);
    void set_genome_properties(RNN_Genome* genome);
    void get_time_series_parameters(TimeSeriesSets* time_series_sets);
    uniform_int_distribution<int32_t> get_recurrent_depth_dist();
};

#endif