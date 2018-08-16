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

#include "mpi.h"

#include "common/arguments.hxx"

#include "rnn/lstm_node.hxx"
#include "rnn/rnn_edge.hxx"
#include "rnn/rnn_genome.hxx"
#include "rnn/rnn_node.hxx"
#include "rnn/rnn_node_interface.hxx"

#include "rnn/generate_nn.hxx"

#include "time_series/time_series.hxx"

bool verbose = false;

int input_length = 10;
int test_iterations = 1000;


minstd_rand0 generator;
uniform_real_distribution<double> rng(-0.5, 0.5);

void generate_random_vector(int number_parameters, vector<double> &v) {
    v.resize(number_parameters);

    for (uint32_t i = 0; i < number_parameters; i++) {
        v[i] = rng(generator);
    }
}

void test_rnn(string name, RNN_Genome *genome, const vector< vector<double> > &inputs, const vector< vector<double> > &outputs) {
    double analytic_mse, empirical_mse;
    vector<double> parameters;
    vector<double> analytic_gradient, empirical_gradient;

    cout << "testing gradient on '" << name << "' ... ";
    bool failed = false;

    RNN* rnn = genome->get_rnn();


    for (uint32_t i = 0; i < test_iterations; i++) {
        if (verbose) {
            if (i == 0) cout << endl;
            cout << "\tAttempt " << i << endl;
        }

        generate_random_vector(rnn->get_number_weights(), parameters);

        rnn->get_analytic_gradient(parameters, inputs, outputs, analytic_mse, analytic_gradient, false, true, 0.0);
        rnn->get_empirical_gradient(parameters, inputs, outputs, empirical_mse, empirical_gradient, false, true, 0.0);


        for (uint32_t j = 0; j < analytic_gradient.size(); j++) {
            double difference = analytic_gradient[j] - empirical_gradient[j];

            if (verbose) {
                cout << "\t\tanalytic gradient[" << j << "]: " << analytic_gradient[j] << ", empirical gradient[" << j << "]: " << empirical_gradient[j] << ", difference: " << difference << endl;
            }

            if (fabs(difference) > 10e-10) {
                failed = true;
                cout << "FAILED on attempt " << i << "!" << endl;
                cout << "analytic gradient[" << j << "]: " << analytic_gradient[j] << ", empirical gradient[" << j << "]: " << empirical_gradient[j] << ", difference: " << difference << endl;
                //exit(1);
            }
        }
    }

    if (!failed) {
        cout << "SUCCESS!" << endl;
    }
}

int main(int argc, char **argv) {
    vector<string> arguments = vector<string>(argv, argv + argc);

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    //seed = 1337;
    generator = minstd_rand0(seed);

    RNN_Genome *genome;

    cout << "TESTING FEED FORWARD" << endl;

    //test feed forward NNs
    vector< vector<double> > inputs(1);
    vector< vector<double> > outputs(1);

    generate_random_vector(input_length, inputs[0]);
    generate_random_vector(input_length, outputs[0]);

    bool test_ff = true;
    bool test_jordan = true;
    bool test_elman = true;
    bool test_lstm = true;

    for (int32_t max_recurrent_depth = 1; max_recurrent_depth <= 5; max_recurrent_depth++) {
        cout << "\n\testing with max recurrent depth: " << max_recurrent_depth << endl;

        if (test_ff) {
            //Test 1 input, 1 output, no hidden
            genome = create_ff(1, 0, 0, 1, max_recurrent_depth);
            test_rnn("FF: 1 Input, 1 Output", genome, inputs, outputs);
            delete genome;

            genome = create_ff(1, 1, 1, 1, max_recurrent_depth);
            //genome->write_graphviz("ff_1_1x1_1.gv");
            test_rnn("FF: 1 Input, 1x1 Hidden, 1 Output", genome, inputs, outputs);
            delete genome;

            genome = create_ff(1, 1, 2, 1, max_recurrent_depth);
            test_rnn("FF: 1 Input, 1x2 Hidden, 1 Output", genome, inputs, outputs);
            delete genome;

            genome = create_ff(1, 2, 2, 1, max_recurrent_depth);
            test_rnn("FF: 1 Input, 2x2 Hidden, 1 Output", genome, inputs, outputs);
            delete genome;



            //Test 2 inputs, 2 outputs, no hidden
            genome = create_ff(2, 0, 0, 2, max_recurrent_depth);

            inputs.resize(2);
            outputs.resize(2);
            generate_random_vector(input_length, inputs[0]);
            generate_random_vector(input_length, outputs[0]);
            generate_random_vector(input_length, inputs[1]);
            generate_random_vector(input_length, outputs[1]);

            test_rnn("FF: 2 Input, 2 Output", genome, inputs, outputs);
            delete genome;


            genome = create_ff(2, 2, 2, 2, max_recurrent_depth);
            test_rnn("FF: 2 Input, 2x2 Hidden, 2 Output", genome, inputs, outputs);
            delete genome;

            genome = create_ff(2, 2, 3, 2, max_recurrent_depth);
            test_rnn("FF: 2 Input, 2x3 Hidden, 2 Output", genome, inputs, outputs);
            delete genome;

            genome = create_ff(2, 3, 3, 2, max_recurrent_depth);
            test_rnn("FF: 2 Input, 3x3 Hidden, 2 Output", genome, inputs, outputs);
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

            test_rnn("FF: Three Input, Three Output", genome, inputs, outputs);
            delete genome;

            genome = create_ff(3, 3, 3, 3, max_recurrent_depth);
            test_rnn("FF: 3 Input, 3x3 Hidden, 3 Output", genome, inputs, outputs);
            delete genome;

            genome = create_ff(3, 3, 4, 3, max_recurrent_depth);
            test_rnn("FF: 3 Input, 3x4 Hidden, 3 Output", genome, inputs, outputs);
            delete genome;

            genome = create_ff(3, 4, 4, 3, max_recurrent_depth);
            test_rnn("FF: 3 Input, 4x4 Hidden, 3 Output", genome, inputs, outputs);
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
            test_rnn("JORDAN: 1 Input, 1 Output", genome, inputs, outputs);
            delete genome;

            genome = create_jordan(1, 1, 1, 1, max_recurrent_depth);
            test_rnn("JORDAN: 1 Input, 1x1 Hidden, 1 Output", genome, inputs, outputs);
            delete genome;

            genome = create_jordan(1, 1, 2, 1, max_recurrent_depth);
            test_rnn("JORDAN: 1 Input, 1x2 Hidden, 1 Output", genome, inputs, outputs);
            delete genome;

            genome = create_jordan(1, 2, 2, 1, max_recurrent_depth);
            test_rnn("JORDAN: 1 Input, 2x2 Hidden, 1 Output", genome, inputs, outputs);
            delete genome;



            //Test 2 inputs, 2 outputs, no hidden
            genome = create_jordan(2, 0, 0, 2, max_recurrent_depth);

            inputs.resize(2);
            outputs.resize(2);
            generate_random_vector(input_length, inputs[0]);
            generate_random_vector(input_length, outputs[0]);
            generate_random_vector(input_length, inputs[1]);
            generate_random_vector(input_length, outputs[1]);

            test_rnn("JORDAN: 2 Input, 2 Output", genome, inputs, outputs);
            delete genome;


            genome = create_jordan(2, 2, 2, 2, max_recurrent_depth);
            test_rnn("JORDAN: 2 Input, 2x2 Hidden, 2 Output", genome, inputs, outputs);
            delete genome;

            genome = create_jordan(2, 2, 3, 2, max_recurrent_depth);
            test_rnn("JORDAN: 2 Input, 2x3 Hidden, 2 Output", genome, inputs, outputs);
            delete genome;

            genome = create_jordan(2, 3, 3, 2, max_recurrent_depth);
            test_rnn("JORDAN: 2 Input, 3x3 Hidden, 2 Output", genome, inputs, outputs);
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

            test_rnn("JORDAN: Three Input, Three Output", genome, inputs, outputs);
            delete genome;

            genome = create_jordan(3, 3, 3, 3, max_recurrent_depth);
            test_rnn("JORDAN: 3 Input, 3x3 Hidden, 3 Output", genome, inputs, outputs);
            delete genome;

            genome = create_jordan(3, 3, 4, 3, max_recurrent_depth);
            test_rnn("JORDAN: 3 Input, 3x4 Hidden, 3 Output", genome, inputs, outputs);
            delete genome;

            genome = create_jordan(3, 4, 4, 3, max_recurrent_depth);
            test_rnn("JORDAN: 3 Input, 4x4 Hidden, 3 Output", genome, inputs, outputs);
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
            test_rnn("ELMAN: 1 Input, 1 Output", genome, inputs, outputs);
            delete genome;

            genome = create_elman(1, 1, 1, 1, max_recurrent_depth);
            test_rnn("ELMAN: 1 Input, 1x1 Hidden, 1 Output", genome, inputs, outputs);
            delete genome;

            genome = create_elman(1, 1, 2, 1, max_recurrent_depth);
            test_rnn("ELMAN: 1 Input, 1x2 Hidden, 1 Output", genome, inputs, outputs);
            delete genome;

            genome = create_elman(1, 2, 2, 1, max_recurrent_depth);
            test_rnn("ELMAN: 1 Input, 2x2 Hidden, 1 Output", genome, inputs, outputs);
            delete genome;



            //Test 2 inputs, 2 outputs, no hidden
            genome = create_elman(2, 0, 0, 2, max_recurrent_depth);

            inputs.resize(2);
            outputs.resize(2);
            generate_random_vector(input_length, inputs[0]);
            generate_random_vector(input_length, outputs[0]);
            generate_random_vector(input_length, inputs[1]);
            generate_random_vector(input_length, outputs[1]);

            test_rnn("ELMAN: 2 Input, 2 Output", genome, inputs, outputs);
            delete genome;


            genome = create_elman(2, 2, 2, 2, max_recurrent_depth);
            test_rnn("ELMAN: 2 Input, 2x2 Hidden, 2 Output", genome, inputs, outputs);
            delete genome;

            genome = create_elman(2, 2, 3, 2, max_recurrent_depth);
            test_rnn("ELMAN: 2 Input, 2x3 Hidden, 2 Output", genome, inputs, outputs);
            delete genome;

            genome = create_elman(2, 3, 3, 2, max_recurrent_depth);
            test_rnn("ELMAN: 2 Input, 3x3 Hidden, 2 Output", genome, inputs, outputs);
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

            test_rnn("ELMAN: Three Input, Three Output", genome, inputs, outputs);
            delete genome;

            genome = create_elman(3, 3, 3, 3, max_recurrent_depth);
            test_rnn("ELMAN: 3 Input, 3x3 Hidden, 3 Output", genome, inputs, outputs);
            delete genome;

            genome = create_elman(3, 3, 4, 3, max_recurrent_depth);
            test_rnn("ELMAN: 3 Input, 3x4 Hidden, 3 Output", genome, inputs, outputs);
            delete genome;

            genome = create_elman(3, 4, 4, 3, max_recurrent_depth);
            test_rnn("ELMAN: 3 Input, 4x4 Hidden, 3 Output", genome, inputs, outputs);
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
            test_rnn("LSTM: 1 Input, 1 Output", genome, inputs, outputs);
            delete genome;

            genome = create_lstm(1, 1, 1, 1, max_recurrent_depth);
            test_rnn("LSTM: 1 Input, 1x1 Hidden, 1 Output", genome, inputs, outputs);
            delete genome;

            genome = create_lstm(1, 1, 2, 1, max_recurrent_depth);
            test_rnn("LSTM: 1 Input, 1x2 Hidden, 1 Output", genome, inputs, outputs);
            delete genome;

            genome = create_lstm(1, 2, 2, 1, max_recurrent_depth);
            test_rnn("LSTM: 1 Input, 2x2 Hidden, 1 Output", genome, inputs, outputs);
            delete genome;



            //Test 2 inputs, 2 outputs, no hidden
            genome = create_lstm(2, 0, 0, 2, max_recurrent_depth);

            inputs.resize(2);
            outputs.resize(2);
            generate_random_vector(input_length, inputs[0]);
            generate_random_vector(input_length, outputs[0]);
            generate_random_vector(input_length, inputs[1]);
            generate_random_vector(input_length, outputs[1]);

            test_rnn("LSTM: 2 Input, 2 Output", genome, inputs, outputs);
            delete genome;


            genome = create_lstm(2, 2, 2, 2, max_recurrent_depth);
            test_rnn("LSTM: 2 Input, 2x2 Hidden, 2 Output", genome, inputs, outputs);
            delete genome;

            genome = create_lstm(2, 2, 3, 2, max_recurrent_depth);
            test_rnn("LSTM: 2 Input, 2x3 Hidden, 2 Output", genome, inputs, outputs);
            delete genome;

            genome = create_lstm(2, 3, 3, 2, max_recurrent_depth);
            test_rnn("LSTM: 2 Input, 3x3 Hidden, 2 Output", genome, inputs, outputs);
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

            test_rnn("LSTM: Three Input, Three Output", genome, inputs, outputs);
            delete genome;

            genome = create_lstm(3, 3, 3, 3, max_recurrent_depth);
            test_rnn("LSTM: 3 Input, 3x3 Hidden, 3 Output", genome, inputs, outputs);
            delete genome;

            genome = create_lstm(3, 3, 4, 3, max_recurrent_depth);
            test_rnn("LSTM: 3 Input, 3x4 Hidden, 3 Output", genome, inputs, outputs);
            delete genome;

            genome = create_lstm(3, 4, 4, 3, max_recurrent_depth);
            test_rnn("LSTM: 3 Input, 4x4 Hidden, 3 Output", genome, inputs, outputs);
            delete genome;
        }
    }
}
