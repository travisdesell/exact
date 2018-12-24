#include <chrono>

#include <fstream>
using std::getline;
using std::ifstream;

#include <iostream>
using std::cout;
using std::cerr;
using std::endl;

#include <random>
using std::minstd_rand0;
using std::uniform_real_distribution;

#include <string>
using std::string;

#include <vector>
using std::vector;

#include "common/arguments.hxx"

#include "rnn/gru_node.hxx"
#include "rnn/rnn_edge.hxx"
#include "rnn/rnn_genome.hxx"
#include "rnn/rnn_node.hxx"
#include "rnn/rnn_node_interface.hxx"

#include "rnn/generate_nn.hxx"

#include "time_series/time_series.hxx"

#include "gradient_test.hxx"

int input_length = 10;

int main(int argc, char **argv) {
    vector<string> arguments = vector<string>(argv, argv + argc);

    initialize_generator();

    RNN_Genome *genome;

    cout << "TESTING GRU" << endl;

    //test feed forward NNs
    vector< vector<double> > inputs(1);
    vector< vector<double> > outputs(1);

    generate_random_vector(input_length, inputs[0]);
    generate_random_vector(input_length, outputs[0]);

    bool verbose = argument_exists(arguments, "--verbose");

    for (int32_t max_recurrent_depth = 1; max_recurrent_depth <= 5; max_recurrent_depth++) {
        cout << "testing with max recurrent depth: " << max_recurrent_depth << endl;

        inputs.resize(1);
        outputs.resize(1);

        generate_random_vector(input_length, inputs[0]);
        generate_random_vector(input_length, outputs[0]);


        //Test 1 input, 1 output, no hidden
        genome = create_gru(1, 0, 0, 1, max_recurrent_depth);
        gradient_test("GRU: 1 Input, 1 Output", genome, inputs, outputs, verbose);
        delete genome;

        genome = create_gru(1, 1, 1, 1, max_recurrent_depth);
        gradient_test("GRU: 1 Input, 1x1 Hidden, 1 Output", genome, inputs, outputs, verbose);
        delete genome;

        genome = create_gru(1, 1, 2, 1, max_recurrent_depth);
        gradient_test("GRU: 1 Input, 1x2 Hidden, 1 Output", genome, inputs, outputs, verbose);
        delete genome;

        genome = create_gru(1, 2, 2, 1, max_recurrent_depth);
        gradient_test("GRU: 1 Input, 2x2 Hidden, 1 Output", genome, inputs, outputs, verbose);
        delete genome;



        //Test 2 inputs, 2 outputs, no hidden
        genome = create_gru(2, 0, 0, 2, max_recurrent_depth);

        inputs.resize(2);
        outputs.resize(2);
        generate_random_vector(input_length, inputs[0]);
        generate_random_vector(input_length, outputs[0]);
        generate_random_vector(input_length, inputs[1]);
        generate_random_vector(input_length, outputs[1]);

        gradient_test("GRU: 2 Input, 2 Output", genome, inputs, outputs, verbose);
        delete genome;


        genome = create_gru(2, 2, 2, 2, max_recurrent_depth);
        gradient_test("GRU: 2 Input, 2x2 Hidden, 2 Output", genome, inputs, outputs, verbose);
        delete genome;

        genome = create_gru(2, 2, 3, 2, max_recurrent_depth);
        gradient_test("GRU: 2 Input, 2x3 Hidden, 2 Output", genome, inputs, outputs, verbose);
        delete genome;

        genome = create_gru(2, 3, 3, 2, max_recurrent_depth);
        gradient_test("GRU: 2 Input, 3x3 Hidden, 2 Output", genome, inputs, outputs, verbose);
        delete genome;



        //Test 3 inputs, 3 outputs, no hidden
        genome = create_gru(3, 0, 0, 3, max_recurrent_depth);

        inputs.resize(3);
        outputs.resize(3);
        generate_random_vector(input_length, inputs[0]);
        generate_random_vector(input_length, outputs[0]);
        generate_random_vector(input_length, inputs[1]);
        generate_random_vector(input_length, outputs[1]);
        generate_random_vector(input_length, inputs[2]);
        generate_random_vector(input_length, outputs[2]);

        gradient_test("GRU: Three Input, Three Output", genome, inputs, outputs, verbose);
        delete genome;

        genome = create_gru(3, 3, 3, 3, max_recurrent_depth);
        gradient_test("GRU: 3 Input, 3x3 Hidden, 3 Output", genome, inputs, outputs, verbose);
        delete genome;

        genome = create_gru(3, 3, 4, 3, max_recurrent_depth);
        gradient_test("GRU: 3 Input, 3x4 Hidden, 3 Output", genome, inputs, outputs, verbose);
        delete genome;

        genome = create_gru(3, 4, 4, 3, max_recurrent_depth);
        gradient_test("GRU: 3 Input, 4x4 Hidden, 3 Output", genome, inputs, outputs, verbose);
        delete genome;
    }
}
