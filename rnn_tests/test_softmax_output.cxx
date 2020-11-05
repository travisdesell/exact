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
#include "rnn/rnn_edge.hxx"
#include "rnn/rnn_genome.hxx"
#include "rnn/rnn_node.hxx"
#include "rnn/rnn_node_interface.hxx"

#include "rnn/generate_nn.hxx"

#include "time_series/time_series.hxx"

#include "gradient_test.hxx"

int main(int argc, char **argv)
{
    vector<string> arguments = vector<string>(argv, argv + argc);

    Log::initialize(arguments);
    Log::set_id("main");

    initialize_generator();

    RNN_Genome *genome;

    Log::info("TESTING FEED FORWARD\n");

    vector<vector<double>> inputs;
    vector<vector<double>> outputs;

    int input_length = 10;
    get_argument(arguments, "--input_length", true, input_length);

    string weight_initialize_string = "random";
    get_argument(arguments, "--weight_initialize", false, weight_initialize_string);

    WeightType weight_initialize;
    weight_initialize = get_enum_from_string(weight_initialize_string);

    if (weight_initialize < 0 || weight_initialize >= NUM_WEIGHT_TYPES - 1)
    {
        Log::fatal("weight initialization method %s is set wrong \n", weight_initialize_string.c_str());
    }

    inputs.resize(1);
    outputs.resize(1);

    generate_random_oneHot_vector(input_length, inputs[0]);
    generate_random_oneHot_vector(input_length, outputs[0]);

    vector<string> inputs1{"input 1"};
    vector<string> outputs1{"output 1"};

    vector<double> parameters;

    //Test 1 input, 1 output, no hidden

    Log::info("****************Using Inputs :: ***************************\n");

    genome = create_lstm(inputs1, 0, 0, outputs1, 1, weight_initialize);

    for (size_t i = 0; i < inputs.size(); i++)
    {
        for (size_t j = 0; j < inputs[i].size(); j++)
        {
            Log::info("%f ",inputs[i][j]);
        }
        Log::info("\n");
        
    }

    Log::info("****************Outputs :: ***************************\n");

    for (size_t i = 0; i < inputs.size(); i++)
    {
        for (size_t j = 0; j < inputs[i].size(); j++)
        {
            Log::info("%f ", inputs[i][j]);
        }
        Log::info("\n");
    }

    RNN *rnn = genome->get_rnn();


    generate_random_vector(rnn->get_number_weights(), parameters);
    rnn->set_weights(parameters);

    Log::info("****************Using Softmax :: ***************************\n");

    rnn->enable_use_regression(false);
    rnn->forward_pass(inputs, false, true, 0.0);
    rnn->calculate_error_softmax(outputs);

    Log::info("****************Using MSE :: ***************************\n");

    rnn->enable_use_regression(true);
    rnn->forward_pass(inputs, false, true, 0.0);
    rnn->calculate_error_mse(outputs);
}

