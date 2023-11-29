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

    inputs.resize(1);
    outputs.resize(1);

    generate_random_vector(input_length, inputs[0]);
    generate_random_vector(input_length, outputs[0]);

    vector<string> inputs1{"input 1"};
    vector<string> outputs1{"output 1"};

    if (hidden_node_type.compare("sin") == 0) {
        Log::info("TESTING SIN!!!\n");
        genome_original = create_sin(inputs1, 1, 2, outputs1, max_recurrent_depth, weight_rules);
    }
    string path = "/Users/jaredmurphy/exact/scripts/node_tests/genome_original.bin";
    genome_original->write_to_file(path);
    RNN_Genome* genome_from_file = new RNN_Genome(path);

    int32_t num_weights = genome_original->get_number_weights();
    vector<double> weights;
    generate_random_vector(num_weights, weights);

    RNN* rnn_original = genome_original->get_rnn();
    RNN* rnn_from_file = genome_from_file->get_rnn();

    double analytic_mse_original, analytic_mse_file;
    vector<double> analytic_gradient_original, analytic_gradient_file;

    rnn_original->get_analytic_gradient(
        weights, inputs, outputs, analytic_mse_original, analytic_gradient_original, false, true, 0.0
    );
    rnn_from_file->get_analytic_gradient(
        weights, inputs, outputs, analytic_mse_file, analytic_gradient_file, false, true, 0.0
    );

    if (rnn_original->get_number_nodes() == rnn_from_file->get_number_nodes()) {
        Log::info("PASS: SAME NODE COUNT!!!\n");
    } else {
        Log::info("FAILURE: DIFFERENT NODE COUNT!!!\n");
    }
    for (int i = 0; i < rnn_original->get_number_nodes(); i++) {
        RNN_Node_Interface* test_node_original = rnn_original->get_node(i);
        RNN_Node_Interface* test_node_from_file = rnn_from_file->get_node(i);
        if ((test_node_original->layer_type == OUTPUT_LAYER) && (test_node_from_file->layer_type == OUTPUT_LAYER)) {
            if (test_node_original->output_values == test_node_from_file->output_values) {
                Log::info("PASS: FORWARD PASS OUTPUTS ARE THE SAME!!!\n");
            } else {
                Log::info("FAILURE: FORWARD PASS OUTPUTS NOT EQUAL!!!\n");
            }
        }
    }

    if (analytic_gradient_original == analytic_gradient_file) {
        Log::info("PASS: GRADIENTS ARE THE SAME!!!\n");
    } else {
        Log::info("FAILURE: GRADIENTS ARE NOT EQUAL!!!\n");
    }

    delete genome_original;
}
