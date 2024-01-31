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

    bool use_burn_in_bp_epoch;
    int32_t burn_in_period = 2048;
    int32_t max_burn_in_cycles = 4;
    double bp_epochs_start = 0.5;
    double burn_in_ratio = 2.0;

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

    int32_t compute_bp_iterations(RNN_Genome* genome);

   public:
    GenomeProperty();

    void generate_genome_property_from_arguments(const vector<string>& arguments);
    void set_genome_properties(RNN_Genome* genome);
    void get_time_series_parameters(TimeSeriesSets* time_series_sets);
    
    uniform_int_distribution<int32_t> get_recurrent_depth_dist();
};

#endif
