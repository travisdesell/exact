#include <chrono>

#include <fstream>
using std::getline;
using std::ifstream;

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

RNN_Genome *genome;
RNN* rnn;
bool using_dropout;
double dropout_probability;


int main(int argc, char **argv) {
    vector<string> arguments = vector<string>(argv, argv + argc);

    Log::initialize(arguments);
    Log::set_id("main");

    TimeSeriesSets *time_series_sets = TimeSeriesSets::generate_from_arguments(arguments);

    int32_t time_offset = 1;
    get_argument(arguments, "--time_offset", true, time_offset);

    time_series_sets->export_training_series(time_offset, training_inputs, training_outputs);
    time_series_sets->export_test_series(time_offset, test_inputs, test_outputs);

    //int number_inputs = time_series_sets->get_number_inputs();
    //int number_outputs = time_series_sets->get_number_outputs();

    int32_t number_hidden_layers;
    get_argument(arguments, "--number_hidden_layers", true, number_hidden_layers);

    int32_t number_hidden_nodes;
    get_argument(arguments, "--number_hidden_nodes", true, number_hidden_nodes);

    int32_t max_input_lags;
    get_argument(arguments, "--max_input_lags", true, max_input_lags);

    int32_t max_recurrent_depth;
    get_argument(arguments, "--max_recurrent_depth", true, max_recurrent_depth);

    string weight_initialize_string = "xavier";
    get_argument(arguments, "--weight_initialize", false, weight_initialize_string);
    WeightType weight_initialize;
    weight_initialize = get_enum_from_string(weight_initialize_string);

    string output_filename;
    get_argument(arguments, "--output_filename", true, output_filename);

    vector<string> input_parameter_names = time_series_sets->get_input_parameter_names();
    vector<string> output_parameter_names = time_series_sets->get_output_parameter_names();

    Log::info("creating jordan neural network with inputs: %d, hidden: %dx%d, outputs: %d, max input lags: %d, max recurrent depth: %d\n", input_parameter_names.size(), number_hidden_layers, number_hidden_nodes, output_parameter_names.size(), max_input_lags, max_recurrent_depth);
    vector<RNN_Node_Interface*> rnn_nodes;
    vector<RNN_Node_Interface*> output_layer;
    vector< vector<RNN_Node_Interface*> > layer_nodes(2 + number_hidden_layers);
    vector<RNN_Edge*> rnn_edges;
    vector<RNN_Recurrent_Edge*> recurrent_edges;

    int node_innovation_count = 0;
    int edge_innovation_count = 0;
    int current_layer = 0;

    for (int32_t i = 0; i < input_parameter_names.size(); i++) {
        RNN_Node *node = new RNN_Node(++node_innovation_count, INPUT_LAYER, current_layer, SIMPLE_NODE, input_parameter_names[i]);
        rnn_nodes.push_back(node);
        layer_nodes[current_layer].push_back(node);
    }
    current_layer++;

    for (int32_t i = 0; i < number_hidden_layers; i++) {
        for (int32_t j = 0; j < number_hidden_nodes; j++) {
            RNN_Node *node = new RNN_Node(++node_innovation_count, HIDDEN_LAYER, current_layer, JORDAN_NODE);
            rnn_nodes.push_back(node);
            layer_nodes[current_layer].push_back(node);

            for (uint32_t k = 0; k < layer_nodes[current_layer - 1].size(); k++) {
                rnn_edges.push_back(new RNN_Edge(++edge_innovation_count, layer_nodes[current_layer - 1][k], node));
                for (uint32_t d = 1; d <= max_input_lags; d++) {
                    recurrent_edges.push_back(new RNN_Recurrent_Edge(++edge_innovation_count, d, layer_nodes[current_layer - 1][k], node));
                 }
            }
        }
        current_layer++;
    }

    for (int32_t i = 0; i < output_parameter_names.size(); i++) {
        RNN_Node *output_node = new RNN_Node(++node_innovation_count, OUTPUT_LAYER, current_layer, SIMPLE_NODE, output_parameter_names[i]);
        output_layer.push_back(output_node);

        rnn_nodes.push_back(output_node);

        for (uint32_t k = 0; k < layer_nodes[current_layer - 1].size(); k++) {
            rnn_edges.push_back(new RNN_Edge(++edge_innovation_count, layer_nodes[current_layer - 1][k], output_node));
        }
    }

    //connect the output node with recurrent edges to each hidden node
    for (uint32_t k = 0; k < output_layer.size(); k++) {
        for (int32_t i = 0; i < number_hidden_layers; i++) {
            for (int32_t j = 0; j < number_hidden_nodes; j++) {
                for (int32_t d = 1; d <= max_recurrent_depth; d++) {
                    recurrent_edges.push_back(new RNN_Recurrent_Edge(++edge_innovation_count, d, output_layer[k], layer_nodes[i + 1][j]));
                }
            }
        }
    }

    RNN_Genome *genome = new RNN_Genome(rnn_nodes, rnn_edges, recurrent_edges, weight_initialize, WeightType::NONE, WeightType::NONE);

    genome->set_parameter_names(time_series_sets->get_input_parameter_names(), time_series_sets->get_output_parameter_names());
    genome->set_normalize_bounds(time_series_sets->get_normalize_type(), time_series_sets->get_normalize_mins(), time_series_sets->get_normalize_maxs(), time_series_sets->get_normalize_avgs(), time_series_sets->get_normalize_std_devs());

    rnn = genome->get_rnn();

    uint32_t number_of_weights = genome->get_number_weights();

    Log::info("RNN has %d weights.\n", number_of_weights);
    vector<double> min_bound(number_of_weights, -1.0);
    vector<double> max_bound(number_of_weights, 1.0);

    vector<double> best_parameters;

    using_dropout = false;

    genome->initialize_randomly();

    int bp_iterations;
    get_argument(arguments, "--bp_iterations", true, bp_iterations);
    genome->set_bp_iterations(bp_iterations, 0);

    double learning_rate = 0.0001;
    get_argument(arguments, "--learning_rate", false, learning_rate);

    genome->set_learning_rate(learning_rate);
    genome->set_nesterov_momentum(true);
    genome->enable_high_threshold(1.0);
    genome->enable_low_threshold(0.05);
    genome->disable_dropout();

    if (argument_exists(arguments, "--log_filename")) {
        string log_filename;
        get_argument(arguments, "--log_filename", false, log_filename);
        genome->set_log_filename(log_filename);
    }

    if (argument_exists(arguments, "--stochastic")) {
        genome->backpropagate_stochastic(training_inputs, training_outputs, test_inputs, test_outputs, false, 30, 100);
    } else {
        genome->backpropagate(training_inputs, training_outputs, test_inputs, test_outputs);
    }

    genome->write_to_file(output_filename);

    genome->get_weights(best_parameters);
    Log::info("best test MSE: %lf\n", genome->get_fitness());
    rnn->set_weights(best_parameters);
    Log::info("TRAINING ERRORS:\n");
    Log::info("MSE: %lf\n", genome->get_mse(best_parameters, training_inputs, training_outputs));
    Log::info("MAE: %lf\n", genome->get_mae(best_parameters, training_inputs, training_outputs));

    Log::info("TEST ERRORS:");
    Log::info("MSE: %lf\n", genome->get_mse(best_parameters, test_inputs, test_outputs));
    Log::info("MAE: %lf\n", genome->get_mae(best_parameters, test_inputs, test_outputs));

    Log::release_id("main");
}
