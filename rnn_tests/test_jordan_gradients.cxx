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

#include "rnn/lstm_node.hxx"
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

    cout << "TESTING FEED FORWARD" << endl;

    //test feed forward NNs
    vector< vector<double> > inputs(1);
    vector< vector<double> > outputs(1);

    generate_random_vector(input_length, inputs[0]);
    generate_random_vector(input_length, outputs[0]);

    bool verbose = true;
    bool test_ff = true;
    bool test_jordan = true;
    bool test_elman = true;
    bool test_lstm = true;

    for (int32_t max_recurrent_depth = 1; max_recurrent_depth <= 5; max_recurrent_depth++) {
        cout << "\n\testing with max recurrent depth: " << max_recurrent_depth << endl;

        if (test_ff) {
            //Test 1 input, 1 output, no hidden
            genome = create_ff(1, 0, 0, 1, max_recurrent_depth);
            gradient_test("FF: 1 Input, 1 Output", genome, inputs, outputs, verbose);
            delete genome;

            genome = create_ff(1, 1, 1, 1, max_recurrent_depth);
            //genome->write_graphviz("ff_1_1x1_1.gv");
            gradient_test("FF: 1 Input, 1x1 Hidden, 1 Output", genome, inputs, outputs, verbose);
            delete genome;

            genome = create_ff(1, 1, 2, 1, max_recurrent_depth);
            gradient_test("FF: 1 Input, 1x2 Hidden, 1 Output", genome, inputs, outputs, verbose);
            delete genome;

            genome = create_ff(1, 2, 2, 1, max_recurrent_depth);
            gradient_test("FF: 1 Input, 2x2 Hidden, 1 Output", genome, inputs, outputs, verbose);
            delete genome;



            //Test 2 inputs, 2 outputs, no hidden
            genome = create_ff(2, 0, 0, 2, max_recurrent_depth);

            inputs.resize(2);
            outputs.resize(2);
            generate_random_vector(input_length, inputs[0]);
            generate_random_vector(input_length, outputs[0]);
            generate_random_vector(input_length, inputs[1]);
            generate_random_vector(input_length, outputs[1]);

            gradient_test("FF: 2 Input, 2 Output", genome, inputs, outputs, verbose);
            delete genome;


            genome = create_ff(2, 2, 2, 2, max_recurrent_depth);
            gradient_test("FF: 2 Input, 2x2 Hidden, 2 Output", genome, inputs, outputs, verbose);
            delete genome;

            genome = create_ff(2, 2, 3, 2, max_recurrent_depth);
            gradient_test("FF: 2 Input, 2x3 Hidden, 2 Output", genome, inputs, outputs, verbose);
            delete genome;

            genome = create_ff(2, 3, 3, 2, max_recurrent_depth);
            gradient_test("FF: 2 Input, 3x3 Hidden, 2 Output", genome, inputs, outputs, verbose);
            delete genome;



            //Test 3 inputs, 3 outputs, no hidden
            genome = create_ff(3, 0, 0, 3, max_recurrent_depth);

            inputs.resize(3);
            outputs.resize(3);
            generate_random_vector(input_length, inputs[0]);
            generate_random_vector(input_length, outputs[0]);
            generate_random_vector(input_length, inputs[1]);
            generate_random_vector(input_length, outputs[1]);
            generate_random_vector(input_length, inputs[2]);
            generate_random_vector(input_length, outputs[2]);

            gradient_test("FF: Three Input, Three Output", genome, inputs, outputs, verbose);
            delete genome;

            genome = create_ff(3, 3, 3, 3, max_recurrent_depth);
            gradient_test("FF: 3 Input, 3x3 Hidden, 3 Output", genome, inputs, outputs, verbose);
            delete genome;

            genome = create_ff(3, 3, 4, 3, max_recurrent_depth);
            gradient_test("FF: 3 Input, 3x4 Hidden, 3 Output", genome, inputs, outputs, verbose);
            delete genome;

            genome = create_ff(3, 4, 4, 3, max_recurrent_depth);
            gradient_test("FF: 3 Input, 4x4 Hidden, 3 Output", genome, inputs, outputs, verbose);
            delete genome;
        }

        if (test_jordan) {

            cout << "TESTING JORDAN" << endl;

            inputs.resize(1);
            outputs.resize(1);

            generate_random_vector(input_length, inputs[0]);
            generate_random_vector(input_length, outputs[0]);


            //Test 1 input, 1 output, no hidden
            genome = create_jordan(1, 0, 0, 1, max_recurrent_depth);
            gradient_test("JORDAN: 1 Input, 1 Output", genome, inputs, outputs, verbose);
            delete genome;

            genome = create_jordan(1, 1, 1, 1, max_recurrent_depth);
            gradient_test("JORDAN: 1 Input, 1x1 Hidden, 1 Output", genome, inputs, outputs, verbose);
            delete genome;

            genome = create_jordan(1, 1, 2, 1, max_recurrent_depth);
            gradient_test("JORDAN: 1 Input, 1x2 Hidden, 1 Output", genome, inputs, outputs, verbose);
            delete genome;

            genome = create_jordan(1, 2, 2, 1, max_recurrent_depth);
            gradient_test("JORDAN: 1 Input, 2x2 Hidden, 1 Output", genome, inputs, outputs, verbose);
            delete genome;



            //Test 2 inputs, 2 outputs, no hidden
            genome = create_jordan(2, 0, 0, 2, max_recurrent_depth);

            inputs.resize(2);
            outputs.resize(2);
            generate_random_vector(input_length, inputs[0]);
            generate_random_vector(input_length, outputs[0]);
            generate_random_vector(input_length, inputs[1]);
            generate_random_vector(input_length, outputs[1]);

            gradient_test("JORDAN: 2 Input, 2 Output", genome, inputs, outputs, verbose);
            delete genome;


            genome = create_jordan(2, 2, 2, 2, max_recurrent_depth);
            gradient_test("JORDAN: 2 Input, 2x2 Hidden, 2 Output", genome, inputs, outputs, verbose);
            delete genome;

            genome = create_jordan(2, 2, 3, 2, max_recurrent_depth);
            gradient_test("JORDAN: 2 Input, 2x3 Hidden, 2 Output", genome, inputs, outputs, verbose);
            delete genome;

            genome = create_jordan(2, 3, 3, 2, max_recurrent_depth);
            gradient_test("JORDAN: 2 Input, 3x3 Hidden, 2 Output", genome, inputs, outputs, verbose);
            delete genome;



            //Test 3 inputs, 3 outputs, no hidden
            genome = create_jordan(3, 0, 0, 3, max_recurrent_depth);

            inputs.resize(3);
            outputs.resize(3);
            generate_random_vector(input_length, inputs[0]);
            generate_random_vector(input_length, outputs[0]);
            generate_random_vector(input_length, inputs[1]);
            generate_random_vector(input_length, outputs[1]);
            generate_random_vector(input_length, inputs[2]);
            generate_random_vector(input_length, outputs[2]);

            gradient_test("JORDAN: Three Input, Three Output", genome, inputs, outputs, verbose);
            delete genome;

            genome = create_jordan(3, 3, 3, 3, max_recurrent_depth);
            gradient_test("JORDAN: 3 Input, 3x3 Hidden, 3 Output", genome, inputs, outputs, verbose);
            delete genome;

            genome = create_jordan(3, 3, 4, 3, max_recurrent_depth);
            gradient_test("JORDAN: 3 Input, 3x4 Hidden, 3 Output", genome, inputs, outputs, verbose);
            delete genome;

            genome = create_jordan(3, 4, 4, 3, max_recurrent_depth);
            gradient_test("JORDAN: 3 Input, 4x4 Hidden, 3 Output", genome, inputs, outputs, verbose);
            delete genome;
        }

        if (test_elman) {
            cout << "TESTING ELMAN" << endl;

            inputs.resize(1);
            outputs.resize(1);

            generate_random_vector(input_length, inputs[0]);
            generate_random_vector(input_length, outputs[0]);


            //Test 1 input, 1 output, no hidden
            genome = create_elman(1, 0, 0, 1, max_recurrent_depth);
            gradient_test("ELMAN: 1 Input, 1 Output", genome, inputs, outputs, verbose);
            delete genome;

            genome = create_elman(1, 1, 1, 1, max_recurrent_depth);
            gradient_test("ELMAN: 1 Input, 1x1 Hidden, 1 Output", genome, inputs, outputs, verbose);
            delete genome;

            genome = create_elman(1, 1, 2, 1, max_recurrent_depth);
            gradient_test("ELMAN: 1 Input, 1x2 Hidden, 1 Output", genome, inputs, outputs, verbose);
            delete genome;

            genome = create_elman(1, 2, 2, 1, max_recurrent_depth);
            gradient_test("ELMAN: 1 Input, 2x2 Hidden, 1 Output", genome, inputs, outputs, verbose);
            delete genome;



            //Test 2 inputs, 2 outputs, no hidden
            genome = create_elman(2, 0, 0, 2, max_recurrent_depth);

            inputs.resize(2);
            outputs.resize(2);
            generate_random_vector(input_length, inputs[0]);
            generate_random_vector(input_length, outputs[0]);
            generate_random_vector(input_length, inputs[1]);
            generate_random_vector(input_length, outputs[1]);

            gradient_test("ELMAN: 2 Input, 2 Output", genome, inputs, outputs, verbose);
            delete genome;


            genome = create_elman(2, 2, 2, 2, max_recurrent_depth);
            gradient_test("ELMAN: 2 Input, 2x2 Hidden, 2 Output", genome, inputs, outputs, verbose);
            delete genome;

            genome = create_elman(2, 2, 3, 2, max_recurrent_depth);
            gradient_test("ELMAN: 2 Input, 2x3 Hidden, 2 Output", genome, inputs, outputs, verbose);
            delete genome;

            genome = create_elman(2, 3, 3, 2, max_recurrent_depth);
            gradient_test("ELMAN: 2 Input, 3x3 Hidden, 2 Output", genome, inputs, outputs, verbose);
            delete genome;



            //Test 3 inputs, 3 outputs, no hidden
            genome = create_elman(3, 0, 0, 3, max_recurrent_depth);

            inputs.resize(3);
            outputs.resize(3);
            generate_random_vector(input_length, inputs[0]);
            generate_random_vector(input_length, outputs[0]);
            generate_random_vector(input_length, inputs[1]);
            generate_random_vector(input_length, outputs[1]);
            generate_random_vector(input_length, inputs[2]);
            generate_random_vector(input_length, outputs[2]);

            gradient_test("ELMAN: Three Input, Three Output", genome, inputs, outputs, verbose);
            delete genome;

            genome = create_elman(3, 3, 3, 3, max_recurrent_depth);
            gradient_test("ELMAN: 3 Input, 3x3 Hidden, 3 Output", genome, inputs, outputs, verbose);
            delete genome;

            genome = create_elman(3, 3, 4, 3, max_recurrent_depth);
            gradient_test("ELMAN: 3 Input, 3x4 Hidden, 3 Output", genome, inputs, outputs, verbose);
            delete genome;

            genome = create_elman(3, 4, 4, 3, max_recurrent_depth);
            gradient_test("ELMAN: 3 Input, 4x4 Hidden, 3 Output", genome, inputs, outputs, verbose);
            delete genome;
        }


        if (test_lstm) {
            cout << "TESTING LSTMS" << endl;

            inputs.resize(1);
            outputs.resize(1);

            generate_random_vector(input_length, inputs[0]);
            generate_random_vector(input_length, outputs[0]);


            //Test 1 input, 1 output, no hidden
            genome = create_lstm(1, 0, 0, 1, max_recurrent_depth);
            gradient_test("LSTM: 1 Input, 1 Output", genome, inputs, outputs, verbose);
            delete genome;

            genome = create_lstm(1, 1, 1, 1, max_recurrent_depth);
            gradient_test("LSTM: 1 Input, 1x1 Hidden, 1 Output", genome, inputs, outputs, verbose);
            delete genome;

            genome = create_lstm(1, 1, 2, 1, max_recurrent_depth);
            gradient_test("LSTM: 1 Input, 1x2 Hidden, 1 Output", genome, inputs, outputs, verbose);
            delete genome;

            genome = create_lstm(1, 2, 2, 1, max_recurrent_depth);
            gradient_test("LSTM: 1 Input, 2x2 Hidden, 1 Output", genome, inputs, outputs, verbose);
            delete genome;



            //Test 2 inputs, 2 outputs, no hidden
            genome = create_lstm(2, 0, 0, 2, max_recurrent_depth);

            inputs.resize(2);
            outputs.resize(2);
            generate_random_vector(input_length, inputs[0]);
            generate_random_vector(input_length, outputs[0]);
            generate_random_vector(input_length, inputs[1]);
            generate_random_vector(input_length, outputs[1]);

            gradient_test("LSTM: 2 Input, 2 Output", genome, inputs, outputs, verbose);
            delete genome;


            genome = create_lstm(2, 2, 2, 2, max_recurrent_depth);
            gradient_test("LSTM: 2 Input, 2x2 Hidden, 2 Output", genome, inputs, outputs, verbose);
            delete genome;

            genome = create_lstm(2, 2, 3, 2, max_recurrent_depth);
            gradient_test("LSTM: 2 Input, 2x3 Hidden, 2 Output", genome, inputs, outputs, verbose);
            delete genome;

            genome = create_lstm(2, 3, 3, 2, max_recurrent_depth);
            gradient_test("LSTM: 2 Input, 3x3 Hidden, 2 Output", genome, inputs, outputs, verbose);
            delete genome;



            //Test 3 inputs, 3 outputs, no hidden
            genome = create_lstm(3, 0, 0, 3, max_recurrent_depth);

            inputs.resize(3);
            outputs.resize(3);
            generate_random_vector(input_length, inputs[0]);
            generate_random_vector(input_length, outputs[0]);
            generate_random_vector(input_length, inputs[1]);
            generate_random_vector(input_length, outputs[1]);
            generate_random_vector(input_length, inputs[2]);
            generate_random_vector(input_length, outputs[2]);

            gradient_test("LSTM: Three Input, Three Output", genome, inputs, outputs, verbose);
            delete genome;

            genome = create_lstm(3, 3, 3, 3, max_recurrent_depth);
            gradient_test("LSTM: 3 Input, 3x3 Hidden, 3 Output", genome, inputs, outputs, verbose);
            delete genome;

            genome = create_lstm(3, 3, 4, 3, max_recurrent_depth);
            gradient_test("LSTM: 3 Input, 3x4 Hidden, 3 Output", genome, inputs, outputs, verbose);
            delete genome;

            genome = create_lstm(3, 4, 4, 3, max_recurrent_depth);
            gradient_test("LSTM: 3 Input, 4x4 Hidden, 3 Output", genome, inputs, outputs, verbose);
            delete genome;
        }
    }
}
