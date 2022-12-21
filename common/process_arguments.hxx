#ifndef PROCESS_ARGUMENTS_HXX
#define PROCESS_ARGUMENTS_HXX

#include <string>
using std::string;

#include <thread>
using std::thread;

#include <vector>
using std::vector;

#include "common/arguments.hxx"
#include "common/log.hxx"
// #include "weights/weight_rules.hxx"
// #include "weights/weight_update.hxx"
#include "examm/island_speciation_strategy.hxx"
#include "examm/neat_speciation_strategy.hxx"
#include "examm/examm.hxx"
// #include "time_series/time_series.hxx"
#include "rnn/rnn_genome.hxx"

EXAMM* generate_examm_from_arguments(const vector<string> &arguments, TimeSeriesSets *time_series_sets, WeightRules *weight_rules, RNN_Genome *seed_genome);
SpeciationStrategy* generate_speciation_strategy_from_arguments(const vector<string> &arguments, RNN_Genome *seed_genome);
IslandSpeciationStrategy* generate_island_speciation_strategy_from_arguments(const vector<string> &arguments, RNN_Genome *seed_genome);
NeatSpeciationStrategy* generate_neat_speciation_strategy_from_arguments(const vector<string> &arguments, RNN_Genome *seed_genome);

bool is_island_strategy (string strategy_name);
bool is_neat_strategy (string strategy_name);
void set_island_transfer_learning_parameters(const vector<string> &arguments, IslandSpeciationStrategy *island_strategy);

void write_time_series_to_file(const vector<string> &arguments, TimeSeriesSets *time_series_sets);
void get_train_validation_data(const vector<string> &arguments, TimeSeriesSets *time_series_sets, vector< vector< vector<double> > > &traing_inputs, vector< vector< vector<double> > > &train_outputs, vector< vector< vector<double> > > &test_inputs, vector< vector< vector<double> > > &test_outputs);
#endif