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
// #include "examm/examm.hxx"
// #include "time_series/time_series.hxx"
#include "rnn/rnn_genome.hxx"

SpeciationStrategy* generate_speciation_strategy_from_arguments(const vector<string> &arguments, RNN_Genome *seed_genome);

IslandSpeciationStrategy* generate_island_speciation_strategy_from_arguments(const vector<string> &arguments, RNN_Genome *seed_genome);
NeatSpeciationStrategy* generate_neat_speciation_strategy_from_arguments(const vector<string> &arguments, RNN_Genome *seed_genome);

bool is_island_strategy (string strategy_name);
bool is_neat_strategy (string strategy_name);

#endif