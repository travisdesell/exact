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

#include "rnn/delta_node.hxx"
#include "rnn/enarc_node.hxx"
#include "rnn/enas_dag_node.hxx"
#include "rnn/gru_node.hxx"
#include "rnn/lstm_node.hxx"
#include "rnn/mgu_node.hxx"
#include "rnn/ugrnn_node.hxx"
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

    Log::info("TESTING DELTA\n");

    vector<vector<double>> inputs;
    vector<vector<double>> outputs;

    int input_length = 10;
    get_argument(arguments, "--input_length", true, input_length);

    int timesteps = 10;
    get_argument(arguments, "--timesteps", true, timesteps);

    string weight_initialize_string = "random";
    get_argument(arguments, "--weight_initialize", false, weight_initialize_string);

    WeightType weight_initialize;
    weight_initialize = get_enum_from_string(weight_initialize_string);

    if (weight_initialize < 0 || weight_initialize >= NUM_WEIGHT_TYPES - 1)
    {
        Log::fatal("weight initialization method %s is set wrong \n", weight_initialize_string.c_str());
    }

    int max_recurrent_depth = 10;
    get_argument(arguments, "--max_recurrent_depth", true, max_recurrent_depth);

    Log::info("testing with max recurrent depth: %d\n", max_recurrent_depth);

    inputs.resize(input_length);
    for (size_t i = 0; i < inputs.size(); i++)
    {
        inputs[i].resize(timesteps);
    }

    outputs.resize(input_length);
    for (size_t i = 0; i < outputs.size(); i++)
    {
        outputs[i].resize(timesteps);
    }

    generate_random_matrix(timesteps, input_length, inputs);
    generate_random_oneHot_matrix(timesteps, input_length, outputs);

    vector<string> inputs1;
    vector<string> outputs1;

    string input_par = "input";
    string output_par = "output";
    for (int32_t i = 0; i < input_length; i++)
    {
        string temp_input_par = input_par + to_string(i);
        string temp_output_par = output_par + to_string(i);
        inputs1.push_back(temp_input_par);
        outputs1.push_back(temp_output_par);
    }

    genome = create_delta(inputs1, 2, 2, outputs1, max_recurrent_depth, weight_initialize);
    gradient_test_class("DELTA: 1 Input, 1 Output", genome, inputs, outputs);
    delete genome;

    genome = create_enarc(inputs1, 2, 2, outputs1, max_recurrent_depth + 1, weight_initialize);
    gradient_test_class("ENARC: 1 Input, 1 Output", genome, inputs, outputs);
    delete genome;


    genome = create_mgu(inputs1, 2, 2, outputs1, max_recurrent_depth + 1, weight_initialize);
    gradient_test_class("MGU: 1 Input, 1 Output", genome, inputs, outputs);
    delete genome;

    genome = create_ugrnn(inputs1, 2, 2, outputs1, max_recurrent_depth + 1, weight_initialize);
    gradient_test_class("UGRNN: 1 Input, 1 Output", genome, inputs, outputs);
    delete genome;


}
