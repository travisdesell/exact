#include <chrono>

#include <fstream>
using std::getline;
using std::ifstream;
using std::ofstream;

#include <random>
using std::minstd_rand0;
using std::uniform_real_distribution;

#include <string>
using std::string;

#include <vector>
using std::vector;

#include "common/arguments.hxx"
#include "common/log.hxx"
#include "common/weight_initialize.hxx"
#include "common/weight_update.hxx"
#include "common/files.hxx"

#include "rnn/lstm_node.hxx"
#include "rnn/gru_node.hxx"
#include "rnn/rnn_edge.hxx"
#include "rnn/rnn_genome.hxx"
#include "rnn/rnn_node.hxx"
#include "rnn/rnn_node_interface.hxx"

#include "rnn/generate_nn.hxx"

#include "time_series/time_series.hxx"

vector< vector< vector<double> > > training_inputs;
vector< vector< vector<double> > > training_outputs;
vector< vector< vector<double> > > test_inputs;
vector< vector< vector<double> > > test_outputs;

bool random_sequence_length;
int32_t sequence_length_lower_bound = 30;
int32_t sequence_length_upper_bound = 100;

RNN_Genome *genome;
RNN* rnn;
WeightUpdate *weight_update_method;
int32_t bp_iterations;
bool using_dropout;
double dropout_probability;

ofstream *log_file;
string output_directory;

double objective_function(const vector<double> &parameters) {
    rnn->set_weights(parameters);

    double error = 0.0;

    for (int32_t i = 0; i < (int32_t)training_inputs.size(); i++) {
        error += rnn->prediction_mae(training_inputs[i], training_outputs[i], false, true, 0.0);
    }

    return -error;
}

double test_objective_function(const vector<double> &parameters) {
    rnn->set_weights(parameters);

    double total_error = 0.0;

    for (int32_t i = 0; i < (int32_t)test_inputs.size(); i++) {
        double error = rnn->prediction_mse(test_inputs[i], test_outputs[i], false, true, 0.0);
        total_error += error;

        Log::info("output for series[%d]: %lf\n", i, error);
    }

    return -total_error;
}


int main(int argc, char **argv) {
    vector<string> arguments = vector<string>(argv, argv + argc);

    Log::initialize(arguments);
    Log::set_id("main");

    TimeSeriesSets *time_series_sets = TimeSeriesSets::generate_from_arguments(arguments);

    int32_t time_offset = 1;
    get_argument(arguments, "--time_offset", true, time_offset);

    time_series_sets->export_training_series(time_offset, training_inputs, training_outputs);
    time_series_sets->export_test_series(time_offset, test_inputs, test_outputs);

    int number_inputs = time_series_sets->get_number_inputs();
    //int number_outputs = time_series_sets->get_number_outputs();

    string rnn_type;
    get_argument(arguments, "--rnn_type", true, rnn_type);

    int32_t num_hidden_layers;
    get_argument(arguments, "--num_hidden_layers", true, num_hidden_layers);

    int32_t max_recurrent_depth;
    get_argument(arguments, "--max_recurrent_depth", true, max_recurrent_depth);

    string weight_initialize_string = "xavier";
    get_argument(arguments, "--weight_initialize", false, weight_initialize_string);
    WeightType weight_initialize;
    weight_initialize = get_enum_from_string(weight_initialize_string);

    weight_update_method = new WeightUpdate();
    weight_update_method->generate_from_arguments(arguments);

    vector<string> input_parameter_names = time_series_sets->get_input_parameter_names();
    vector<string> output_parameter_names = time_series_sets->get_output_parameter_names();

    RNN_Genome *genome;
    Log::info("RNN TYPE = %s\n", rnn_type.c_str());
    if (rnn_type == "lstm") {
        genome = create_lstm(input_parameter_names, num_hidden_layers, number_inputs, output_parameter_names, max_recurrent_depth, weight_initialize);

    } else if (rnn_type == "gru") {
        genome = create_gru(input_parameter_names, num_hidden_layers, number_inputs, output_parameter_names, max_recurrent_depth, weight_initialize);

    } else if (rnn_type == "delta") {
        genome = create_delta(input_parameter_names, num_hidden_layers, number_inputs, output_parameter_names, max_recurrent_depth, weight_initialize);

    } else if (rnn_type == "mgu") {
        genome = create_mgu(input_parameter_names, num_hidden_layers, number_inputs, output_parameter_names, max_recurrent_depth, weight_initialize);

    } else if (rnn_type == "ugrnn") {
        genome = create_ugrnn(input_parameter_names, num_hidden_layers, number_inputs, output_parameter_names, max_recurrent_depth, weight_initialize);

    } else if (rnn_type == "ff") {
        genome = create_ff(input_parameter_names, num_hidden_layers, number_inputs, output_parameter_names, max_recurrent_depth, weight_initialize, WeightType::NONE, WeightType::NONE);

    } else if (rnn_type == "jordan") {
        genome = create_jordan(input_parameter_names, num_hidden_layers, number_inputs, output_parameter_names, max_recurrent_depth, weight_initialize);

    } else if (rnn_type == "elman") {
        genome = create_elman(input_parameter_names, num_hidden_layers, number_inputs, output_parameter_names, max_recurrent_depth, weight_initialize);
    } else if (rnn_type == "dnas") {
        vector<int> node_types = { SIMPLE_NODE, SIMPLE_NODE };
        genome = create_dnas_nn(input_parameter_names, 4, 12, output_parameter_names, max_recurrent_depth, node_types, weight_initialize);
    } else {
        Log::fatal("ERROR: incorrect rnn type\n");
        Log::fatal("Possibilities are:\n");
        Log::fatal("    lstm\n");
        Log::fatal("    gru\n");
        Log::fatal("    ff\n");
        Log::fatal("    jordan\n");
        Log::fatal("    elman\n");        
        exit(1);
    }

    get_argument(arguments, "--bp_iterations", true, bp_iterations);
    genome->set_bp_iterations(bp_iterations, 0);

    get_argument(arguments, "--output_directory", true, output_directory);
    if (output_directory != "") {
        mkpath(output_directory.c_str(), 0777);
    }
    if (argument_exists(arguments, "--log_filename")) {
        string log_filename;
        get_argument(arguments, "--log_filename", true, log_filename);
        genome->set_log_filename(output_directory + "/" + log_filename);
    } 

    genome->set_parameter_names(time_series_sets->get_input_parameter_names(), time_series_sets->get_output_parameter_names());
    genome->set_normalize_bounds(time_series_sets->get_normalize_type(), time_series_sets->get_normalize_mins(), time_series_sets->get_normalize_maxs(), time_series_sets->get_normalize_avgs(), time_series_sets->get_normalize_std_devs());

    rnn = genome->get_rnn();

    int32_t number_of_weights = genome->get_number_weights();

    Log::info("RNN has %d weights.\n", number_of_weights);
    vector<double> min_bound(number_of_weights, -1.0);
    vector<double> max_bound(number_of_weights, 1.0);

    vector<double> best_parameters;

    using_dropout = false;

    genome->initialize_randomly();

    double learning_rate = 0.001;
    get_argument(arguments, "--learning_rate", false, learning_rate);

    genome->set_learning_rate(learning_rate);
    // genome->set_nesterov_momentum(true);
    genome->enable_high_threshold(1.0);
    genome->enable_low_threshold(0.05);
    genome->disable_dropout();
    // genome->enable_use_regression(true);

    random_sequence_length = argument_exists(arguments, "--random_sequence_length");
    get_argument(arguments, "--sequence_length_lower_bound", false, sequence_length_lower_bound);
    get_argument(arguments, "--sequence_length_upper_bound", false, sequence_length_upper_bound);

    if (argument_exists(arguments, "--stochastic")) {
        Log::info("running stochastic back prop \n");
        genome->backpropagate_stochastic(training_inputs, training_outputs, test_inputs, test_outputs, random_sequence_length, sequence_length_lower_bound, sequence_length_upper_bound, weight_update_method);
    } else {
        genome->backpropagate(training_inputs, training_outputs, test_inputs, test_outputs, weight_update_method);
    }

    Log::info("Training finished\n");
    genome->get_weights(best_parameters);
    rnn->set_weights(best_parameters);

    Log::info("TRAINING ERRORS:\n");
    Log::info("MSE: %lf\n", genome->get_mse(best_parameters, training_inputs, training_outputs));
    Log::info("MAE: %lf\n", genome->get_mae(best_parameters, training_inputs, training_outputs));

    Log::info("TEST ERRORS:\n");
    Log::info("MSE: %lf\n", genome->get_mse(best_parameters, test_inputs, test_outputs));
    Log::info("MAE: %lf\n", genome->get_mae(best_parameters, test_inputs, test_outputs));

    Log::release_id("main");
}
