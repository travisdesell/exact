#ifndef RNN_GENERATE_NN_HXX
#define RNN_GENERATE_NN_HXX

#include <string>
using std::string;

#include <vector>
using std::vector;

#include "rnn/rnn_genome.hxx"
#include "weights/weight_rules.hxx"
#include "common/arguments.hxx"

RNN_Genome* create_ff(const vector<string> &input_parameter_names, int32_t number_hidden_layers, int32_t number_hidden_nodes, const vector<string> &output_parameter_names, int32_t max_recurrent_depth, WeightRules *weight_rules);

RNN_Genome* create_jordan(const vector<string> &input_parameter_names, int32_t number_hidden_layers, int32_t number_hidden_nodes, const vector<string> &output_parameter_names, int32_t max_recurrent_depth, WeightRules *weight_rules);

RNN_Genome* create_elman(const vector<string> &input_parameter_names, int32_t number_hidden_layers, int32_t number_hidden_nodes, const vector<string> &output_parameter_names, int32_t max_recurrent_depth, WeightRules *weight_rules);

RNN_Genome* create_lstm(const vector<string> &input_parameter_names, int32_t number_hidden_layers, int32_t number_hidden_nodes, const vector<string> &output_parameter_names, int32_t max_recurrent_depth, WeightRules *weight_rules);

RNN_Genome* create_delta(const vector<string> &input_parameter_names, int32_t number_hidden_layers, int32_t number_hidden_nodes, const vector<string> &output_parameter_names, int32_t max_recurrent_depth, WeightRules *weight_rules);

RNN_Genome* create_gru(const vector<string> &input_parameter_names, int32_t number_hidden_layers, int32_t number_hidden_nodes, const vector<string> &output_parameter_names, int32_t max_recurrent_depth, WeightRules *weight_rules);

RNN_Genome* create_enarc(const vector<string> &input_parameter_names, int32_t number_hidden_layers, int32_t number_hidden_nodes, const vector<string> &output_parameter_names, int32_t max_recurrent_depth, WeightRules *weight_rules);

RNN_Genome* create_enas_dag(const vector<string> &input_parameter_names, int32_t number_hidden_layers, int32_t number_hidden_nodes, const vector<string> &output_parameter_names, int32_t max_recurrent_depth, WeightRules *weight_rules);

RNN_Genome* create_random_dag(const vector<string> &input_parameter_names, int32_t number_hidden_layers, int32_t number_hidden_nodes, const vector<string> &output_parameter_names, int32_t max_recurrent_depth, WeightRules *weight_rules);

RNN_Genome* create_mgu(const vector<string> &input_parameter_names, int32_t number_hidden_layers, int32_t number_hidden_nodes, const vector<string> &output_parameter_names, int32_t max_recurrent_depth, WeightRules *weight_rules);

RNN_Genome* create_ugrnn(const vector<string> &input_parameter_names, int32_t number_hidden_layers, int32_t number_hidden_nodes, const vector<string> &output_parameter_names, int32_t max_recurrent_depth, WeightRules *weight_rules);

RNN_Genome* get_seed_genome(const vector<string> &arguments, TimeSeriesSets *time_series_sets, WeightRules *weight_rules);

#endif
