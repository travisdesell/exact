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

#include <iostream>

#include "common/arguments.hxx"
#include "common/log.hxx"
#include "gradient_test.hxx"
#include "rnn/generate_nn.hxx"
#include "rnn/rnn_edge.hxx"
#include "rnn/rnn_genome.hxx"
#include "rnn/rnn_node.hxx"
#include "rnn/rnn_node_interface.hxx"
#include "rnn/sin_node.hxx"
#include "time_series/time_series.hxx"
#include "weights/weight_rules.hxx"

int main(int argc, char** argv) {
    vector<string> arguments = vector<string>(argv, argv + argc);

    Log::initialize(arguments);
    Log::set_id("main");

    initialize_generator();

    RNN_Genome* genome_original = nullptr;

    vector<vector<double> > inputs;
    vector<vector<double> > outputs;

    int input_length = 10;
    string hidden_node_type;
    get_argument(arguments, "--hidden_node_type", true, hidden_node_type);

    WeightRules* weight_rules = new WeightRules();

    int32_t max_recurrent_depth = 3;
    Log::info("testing with max recurrent depth: %d\n", max_recurrent_depth);

    inputs.resize(3);
    outputs.resize(3);

    vector<string> inputs3{"input 1", "input 2", "input 3"};
    vector<string> outputs3{"output 1", "output 2", "input 3"};
    Log::info("testing with 3 input nodes, 3 output nodes\n");

    generate_random_vector(input_length, inputs[0]);
    generate_random_vector(input_length, outputs[0]);
    generate_random_vector(input_length, inputs[1]);
    generate_random_vector(input_length, outputs[1]);
    generate_random_vector(input_length, inputs[2]);
    generate_random_vector(input_length, outputs[2]);

    

    if (hidden_node_type.compare("sin") == 0) {
        Log::info("TESTING SIN!!!\n");
        genome_original = create_sin(inputs3, 1, 5, outputs3, max_recurrent_depth, weight_rules);
        Log::info("testing with 1 hidden layer, 5 sin nodes\n");
    }
    
    int32_t num_weights = genome_original->get_number_weights();
    vector<double> best_parameters_original;
    generate_random_vector(num_weights, best_parameters_original);
    genome_original->set_best_parameters(best_parameters_original);
    
    
    string path = "./genome_original.bin";
    genome_original->write_to_file(path);
    RNN_Genome* genome_file = new RNN_Genome(path);
    vector<double> best_parameters_file = genome_file->get_best_parameters();

    if (best_parameters_original == best_parameters_file){
        Log::info("PASS: BEST PARAMETERS ARE EQUAL!!!\n");
    } else {
        Log::info("FAILURE: BEST PARAMETERS ARE NOT EQUAL!!!\n");
    }
              
    RNN* rnn_original = genome_original->get_rnn();
    RNN* rnn_file = genome_file->get_rnn();

    double analytic_mse_original, analytic_mse_file;
    vector<double> analytic_gradient_original, analytic_gradient_file;

    rnn_original->get_analytic_gradient(
        best_parameters_original, inputs, outputs, analytic_mse_original, analytic_gradient_original, false, true, 0.0
    );
    rnn_file->get_analytic_gradient(
        best_parameters_file, inputs, outputs, analytic_mse_file, analytic_gradient_file, false, true, 0.0
    );

    if (rnn_original->get_number_nodes() == rnn_file->get_number_nodes()) {
        Log::info("PASS: SAME NODE COUNT!!!\n");
    } else {
        Log::info("FAILURE: DIFFERENT NODE COUNT!!!\n");
    }
    if (rnn_original->get_number_edges() == rnn_file->get_number_edges()) {
        Log::info("PASS: SAME EDGE COUNT!!!\n");
    } else {
        Log::info("FAILURE: DIFFERENT EDGE COUNT!!!\n");
    }
    int output_node_count = 1;
    for (int i = 0; i < rnn_original->get_number_nodes(); i++) {
        RNN_Node_Interface* test_node_original = rnn_original->get_node(i);
        RNN_Node_Interface* test_node_file = rnn_file->get_node(i);
        if ((test_node_original->layer_type == OUTPUT_LAYER) && (test_node_file->layer_type == OUTPUT_LAYER)) {
            if (test_node_original->output_values == test_node_file->output_values) {
                Log::info("PASS: FORWARD PASS OUTPUT NODE %d - OUTPUT IS THE SAME!!!\n", output_node_count);
            } else {
                Log::info("FAILURE: FORWARD PASS OUTPUT NODE %d - OUTPUT IS NOT EQUAL!!!\n", output_node_count);
            }
            ++output_node_count;
        }
    }

    if (analytic_gradient_original == analytic_gradient_file) {
        Log::info("PASS: GRADIENTS ARE THE SAME!!!\n");
    } else {
        Log::info("FAILURE: GRADIENTS ARE NOT EQUAL!!!\n");
    }

    delete genome_original;
    delete genome_file;
}
