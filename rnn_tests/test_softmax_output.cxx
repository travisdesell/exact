#include <chrono>

#include <fstream>
using std::getline;
using std::ifstream;

#include <random>
using std::minstd_rand0;
using std::uniform_int_distribution;
using std::uniform_real_distribution;

#include <string>
using std::string;
using std::to_string;

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

    string weight_initialize_string = "random";
    get_argument(arguments, "--weight_initialize", false, weight_initialize_string);

    WeightType weight_initialize;
    weight_initialize = get_enum_from_string(weight_initialize_string);

    if (weight_initialize < 0 || weight_initialize >= NUM_WEIGHT_TYPES - 1)
    {
        Log::fatal("weight initialization method %s is set wrong \n", weight_initialize_string.c_str());
    }

    Log::info("TESTING FEED FORWARD\n");

    vector<vector<double>> inputs;
    vector<vector<double>> outputs;

    int timesteps = 10;
    get_argument(arguments, "--timestep", true, timesteps);

    int input_length = 1;
    get_argument(arguments,"--input_length",true,input_length);

    inputs.resize(input_length);
    for (size_t i = 0; i < inputs.size(); i++) {
        inputs[i].resize(timesteps);
    }

    outputs.resize(input_length);
    for (size_t i = 0; i < outputs.size(); i++) {
        outputs[i].resize(timesteps);
    }

    generate_random_oneHot_matrix(timesteps,input_length,inputs);
    generate_random_oneHot_matrix(timesteps, input_length, outputs);

    vector<double> parameters;
    double mse;
    vector<vector<double>> softmax_gradients, deltas;


    vector<string> inputs1;
    vector<string> outputs1;

    string input_par = "input";
    string output_par = "output";
    for (int32_t i = 0; i < input_length; i++) {
        string temp_input_par = input_par + to_string(i);
        string temp_output_par = output_par + to_string(i);
        inputs1.push_back(temp_input_par);
        outputs1.push_back(temp_output_par);
    }


    genome = create_lstm(inputs1, 1, 1, outputs1, 3, weight_initialize);

    RNN *rnn = genome->get_rnn();

    generate_random_vector(rnn->get_number_weights(), parameters);
    rnn->set_weights(parameters);
    rnn->enable_use_regression(false);
    deltas = rnn->get_softmax_gradient(parameters, inputs, outputs, mse, softmax_gradients, false, true, 0.0);


    for (uint32_t j = 0; j < deltas[0].size(); j++) {
        for (uint32_t i = 0; i < deltas.size(); i++) {
            double difference = softmax_gradients[i][j] - deltas[i][j];
            if (fabs(difference) > 10e-10)
            {
                Log::info("\t\tFAILED softmax gradient[%d][%d]: %lf, deltas[%d][%d]: %lf, difference: %lf\n",i, j, softmax_gradients[i][j],i, j, deltas[i][j], difference);
                //exit(1);
            }
            else
            {
                Log::info("\t\tPASSED softmax gradient[%d][%d]: %lf, deltas[%d][%d]: %lf, difference: %lf\n", i, j, softmax_gradients[i][j], i, j, deltas[i][j], difference);
            }
        }
    }
}

